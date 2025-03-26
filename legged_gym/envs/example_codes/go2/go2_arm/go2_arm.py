from legged_gym.envs.go2.go2_arm.go2_arm_config import Go2ArmCfg
from legged_gym.envs.go2.go2_torque.go2_torque import GO2Torque
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymapi
from isaacgym import gymtorch
import torch


class Go2Arm(GO2Torque):
    cfg = Go2ArmCfg

    def __init__(self, cfg: Go2ArmCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg = cfg
        self.device = sim_device
        self.num_envs = self.cfg.env.num_envs
        self.load_low_level_controller()
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.activation_sign = torch.zeros(self.cfg.env.num_envs, 12, device=self.device)
        self.base_pos = self.root_states[:, :3].clone()
        self.robot_init_pos = torch.zeros(self.num_envs, 2, device=self.device)  # 机器人初始 xy 位置
        self.target_pos = torch.tensor(self.cfg.env.target_pos[:2], device=self.device, dtype=torch.float32)
        self.target_relative_pos = torch.zeros(self.num_envs, 2, device=self.device)
        self.target_distance = torch.zeros(self.num_envs, 1, device=self.device)
        self.target_direction = torch.zeros(self.num_envs, 2, device=self.device)



    def _init_buffers(self):
        super()._init_buffers()
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device,
                                   requires_grad=False)

    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 机器人翻倒检测
        flipped = self.projected_gravity[:, 2] > 0
        self.reset_buf |= flipped
        if flipped.any():
            print("机器人翻倒，环境提前终止")

        # 机器人走出边界
        out_of_bounds = (self.base_pos[:, 0] < -10) | (self.base_pos[:, 0] > 10) | \
                        (self.base_pos[:, 1] < -10) | (self.base_pos[:, 1] > 10)
        self.reset_buf |= out_of_bounds
        if out_of_bounds.any():
            print("机器人超出边界，环境提前终止")

        # 目标达成
        reached_target = self.target_distance.squeeze() < 0.2
        self.reset_buf |= reached_target
        if reached_target.any():
            print("机器人成功到达目标点")

        # 时间步超限
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        if self.time_out_buf.any():
            print("训练达到最大时间步")

        return self.reset_buf

    def load_low_level_controller(self):
        self.low_level_controller = torch.jit.load(self.cfg.low_level_policy.policy_path).to(self.device)
        self.low_level_obs_buf = torch.zeros(self.num_envs, self.cfg.low_level_policy.cfg.env.num_observations,
                                             dtype=torch.float, device=self.device, requires_grad=False)
        self.low_level_actions = torch.zeros(self.num_envs, self.cfg.low_level_policy.cfg.env.num_actions,
                                             dtype=torch.float, device=self.device, requires_grad=False)
        self.low_level_obs_scales = self.cfg.low_level_policy.cfg.normalization.obs_scales
        self.low_level_action_scales = self.cfg.low_level_policy.cfg.control.action_scale
        self.low_level_controller.eval()
        # noise
        self.low_level_noise = torch.zeros_like(self.low_level_obs_buf[0])
        self.low_level_add_noise = self.cfg.low_level_policy.cfg.noise.add_noise
        noise_scales = self.cfg.low_level_policy.cfg.noise.noise_scales
        noise_level = self.cfg.low_level_policy.cfg.noise.noise_level
        self.low_level_noise[:3] = noise_scales.lin_vel * noise_level * self.low_level_obs_scales.lin_vel
        self.low_level_noise[3:6] = noise_scales.ang_vel * noise_level * self.low_level_obs_scales.ang_vel
        self.low_level_noise[6:9] = noise_scales.gravity * noise_level
        self.low_level_noise[9:21] = noise_scales.dof_pos * noise_level * self.low_level_obs_scales.dof_pos
        self.low_level_noise[21:33] = noise_scales.dof_vel * noise_level * self.low_level_obs_scales.dof_vel
        self.low_level_noise[33:38] = 0.  # commands
        self.low_level_noise[38:49] = 0.  # actions
        self.low_level_noise[50:62] = noise_scales.fatigue * noise_level / 10

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[62 + 8:62 + 16] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos # arm joint*8的last action和位置
        noise_vec[62 + 16:62 + 24] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel # 8 joints vel
        noise_vec[86:88] = noise_scales.target_pos * noise_level # 目标相对位置 (dx, dy) 的噪声

        # print("noise_vec.shape:", noise_vec.shape)  # 打印长度
        # print("self.obs_buf.shape:", self.obs_buf.shape)  # 打印 obs_buf 长度

        return noise_vec

    def update_target_relative_pos(self):
        # 获取机器人在世界坐标系下的位置
        robot_x = self.base_pos[:, 0]  # 世界坐标 X
        robot_y = self.base_pos[:, 1]  # 世界坐标 Y

        # 获取目标点在世界坐标系下的位置
        target_x = self.commands[:, 0]
        target_y = self.commands[:, 1]

        # 计算目标点相对于机器人在世界坐标系下的偏移量
        dx = target_x - robot_x
        dy = target_y - robot_y

        # 获取机器人的航向角（yaw 角） 🔄
        robot_yaw = torch.atan2(
            2.0 * (self.base_quat[:, 3] * self.base_quat[:, 2] + self.base_quat[:, 0] * self.base_quat[:, 1]),
            1.0 - 2.0 * (self.base_quat[:, 1] ** 2 + self.base_quat[:, 2] ** 2))  # 计算 yaw 角

        # 旋转目标点到局部坐标系（local frame）
        local_dx = torch.cos(robot_yaw) * dx + torch.sin(robot_yaw) * dy
        local_dy = -torch.sin(robot_yaw) * dx + torch.cos(robot_yaw) * dy

        # 更新 target_relative_pos
        self.target_relative_pos = torch.stack((local_dx, local_dy), dim=-1)

        # print(f"目标点在世界坐标系: {(target_x.mean().item(), target_y.mean().item())}")
        # print(f"机器人在世界坐标系: {(robot_x.mean().item(), robot_y.mean().item())}")
        # print(f"目标点在机器人局部坐标系: {(local_dx.mean().item(), local_dy.mean().item())}")

    def clamp_actions(self):
        """限制动作范围到低级控制器可接受的范围"""
        action_min = torch.tensor([-0.5, -0.5, -1.5, -3.14, -0.5, 0.1], device=self.device)
        action_max = torch.tensor([1.5, 0.5, 1.5, 3.14, 0.5, 0.5], device=self.device)
        self.actions[:, :6] = torch.clamp(self.actions[:, :6], action_min, action_max)

    def compute_low_level_observations(self):
        base_lin_vel = self.base_lin_vel
        # TODO: try to limit your action[:, :6] within the range of your low level control accept
        self.clamp_actions()  # 限制动作范围
        low_level_obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                                       self.base_ang_vel * self.obs_scales.ang_vel,
                                       self.projected_gravity,
                                       ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, :12],
                                       (self.dof_vel * self.obs_scales.dof_vel)[:, :12],
                                       self.actions[:, :3] * self.commands_scale[:3],
                                       self.actions[:, 4:5] * self.commands_scale[4:5],
                                       self.actions[:, 5:6] * self.commands_scale[5:6],
                                       self.torques[:, :12],
                                       self.motor_fatigue[:, :12],
                                       ), dim=-1)

        if self.low_level_add_noise:
            low_level_obs_buf += (2 * torch.rand_like(low_level_obs_buf) - 1) * self.low_level_noise

        self.low_level_obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            low_level_obs_buf, self.low_level_obs_buf)


    def compute_observations(self):
        obs_buf = torch.cat((
            self.low_level_obs_buf,
            self.actions[:, 5:],
            ((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, 12:],
            (self.dof_vel * self.obs_scales.dof_vel)[:, 12:],
            self.target_relative_pos,  # 机器人相对目标的位置 dx, dy
            # self.target_direction,  # 归一化目标方向
        ), dim=-1)

        # # 🔹 **打印所有 obs 维度**
        # print(f"low_level_obs_buf: {self.low_level_obs_buf.shape}")
        # print(f"actions[:, 5:]: {self.actions[:, 5:].shape}")
        # print(f"dof_pos: {((self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos)[:, 12:].shape}")
        # print(f"dof_vel: {(self.dof_vel * self.obs_scales.dof_vel)[:, 12:].shape}")
        # print(f"target_relative_pos: {self.target_relative_pos.shape}")
        # # print(f"target_direction: {self.target_direction.shape}")  # 如果有归一化方向
        #
        # print(f"obs_buf.shape: {obs_buf.shape}")

        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf
        )

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions = actions.to(self.device)
        self.compute_low_level_observations()
        self.low_level_actions = self.low_level_controller(self.low_level_obs_buf.detach()).detach()
        cat_actions = torch.cat((self.low_level_actions, self.actions[:, 5:]), dim=1)
        # step physics and render each frame
        self.render()
        self.rew_buf[:] = 0.
        while self.current_dt * self.current_freq < 1:
            self.torques = self._compute_torques(cat_actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.current_dt += self.dt
            self.post_physics_step()
        self.current_dt %= (1 / self.current_freq)

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ Check terminations, compute observations and rewards
            Calls self._post_physics_step_callback() for common computations
            Calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # 更新目标点的相对位置
        self.update_target_relative_pos()
        self.target_distance = torch.norm(self.target_relative_pos, dim=-1, keepdim=True)

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_base_lin_vel[:] = self.base_lin_vel[:]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        #
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(
            as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

    def _compute_torques(self, actions):
        self._update_growth_scale()
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        self.torques_action = actions_scaled

        # 确保 self.torque_limits 是 2D
        if len(self.torque_limits.shape) == 1:
            self.torque_limits = self.torque_limits.view(1, -1)  # 变成 (1, 20)

        # 确保 self.dof_vel_limits 也是 2D
        if len(self.dof_vel_limits.shape) == 1:
            self.dof_vel_limits = self.dof_vel_limits.view(1, -1)  # 变成 (1, 20)

        # 计算主关节的扭矩
        if self.cfg.control.hill_model:
            torques = self.activation_sign * self.torque_limits[:, :12] * (
                    1 - torch.sign(self.activation_sign) * self.dof_vel[:, :12] / self.dof_vel_limits[:, :12])
        else:
            torques = self.activation_sign * self.torque_limits[:, :12]

        # PD 控制手臂
        Kp = torch.tensor([5.0] * 8, device=self.device)  # P增益
        Kd = torch.tensor([0.5] * 8, device=self.device)  # D增益
        arm_target_pos = self.default_dof_pos[:, 12:]  # 目标关节角度
        arm_current_pos = self.dof_pos[:, 12:]  # 当前关节角度
        arm_current_vel = self.dof_vel[:, 12:]  # 当前关节速度

        torques_for_arm = Kp * (arm_target_pos - arm_current_pos) + Kd * (0 - arm_current_vel)

        # 限制扭矩范围，防止溢出
        torques_for_arm = torch.clamp(torques_for_arm, -self.torque_limits[:, 12:], self.torque_limits[:, 12:])

        # 合并主关节和手臂关节的扭矩
        self.torques = torch.cat((torques, torques_for_arm), dim=1)

        return self.torques

    def _resample_commands(self, env_ids):
        """ 重新采样目标点，使其相对机器人初始位置的范围在 ±10 以内 """
        if len(env_ids) == 0:
            return  # 没有环境需要更新，直接返回

        # 计算目标点的新坐标
        self.commands[env_ids, :2] = (
                self.base_pos[env_ids, :2] + (torch.rand((len(env_ids), 2), device=self.device) * 20 - 10)
        )

        # 调用 update_target_relative_pos 计算局部坐标
        self.update_target_relative_pos()

        ##############################################################################################################

    def _reward_reach_target(self):
        """奖励机器人接近目标点 (x, y)"""
        if not self.cfg.commands.target_command:
            return torch.zeros(self.num_envs, device=self.device)
            # 计算目标距离

        print(f"当前目标距离: {self.target_distance.mean().item()}")

        reward = torch.exp(
            -self.target_distance.squeeze() / self.cfg.rewards.target_tracking_sigma) * self.general_scale

        return reward

