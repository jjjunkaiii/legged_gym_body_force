from legged_gym.envs import LeggedRobot
from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO
from legged_gym.utils.math import wrap_to_pi
from isaacgym.torch_utils import *
from isaacgym import gymtorch
from isaacgym import gymapi
import torch


def update_com(I_box, mass_box, com_box, mass_point, point_pos):
    """
    更新质心位置，计算新质心坐标。

    参数:
    I_box (np.array): 原长方体的惯性矩阵 (3x3)
    mass_box (float): 原长方体的质量
    com_box (np.array): 原长方体的质心位置 (3,)
    mass_point (float): 质点的质量
    point_pos (np.array): 质点的位置 (3,)

    返回:
    new_com (np.array): 新的质心位置
    """
    # 新质心位置公式
    new_com = (mass_box * com_box + mass_point * point_pos) / (mass_box + mass_point)
    return new_com


def parallel_axis_theorem(I_com, mass, d):
    """
    根据平行轴定理计算移动质心后的惯性矩阵

    参数:
    I_com (np.array): 质心处的惯性矩阵 (3x3)
    mass (float): 物体的质量
    d (np.array): 质心相对于新原点的位移 (3,)

    返回:
    np.array: 质心移动后的惯性矩阵 (3x3)
    """
    d_x, d_y, d_z = d
    d_squared = np.array([
        [d_y ** 2 + d_z ** 2, -d_x * d_y, -d_x * d_z],
        [-d_x * d_y, d_x ** 2 + d_z ** 2, -d_y * d_z],
        [-d_x * d_z, -d_y * d_z, d_x ** 2 + d_y ** 2]
    ])

    # 使用平行轴定理计算新的惯性矩阵
    return I_com + mass * d_squared


def update_inertia(I_box, mass_box, com_box, mass_point, point_pos):
    """
    更新惯性矩阵，计算增加质点后的新的惯性矩阵

    参数:
    I_box (np.array): 原长方体的惯性矩阵 (3x3)
    mass_box (float): 原长方体的质量
    com_box (np.array): 原长方体的质心位置 (3,)
    mass_point (float): 质点的质量
    point_pos (np.array): 质点的位置 (3,)

    返回:
    I_total (np.array): 新的惯性矩阵 (3x3)
    new_com (np.array): 新的质心位置 (3,)
    """
    # 更新质心
    new_com = update_com(I_box, mass_box, com_box, mass_point, point_pos)

    # 长方体移动后的惯性矩阵
    displacement_box = com_box - new_com  # 计算长方体质心相对新质心的位移
    I_box_new = parallel_axis_theorem(I_box, mass_box, displacement_box)

    # 质点的惯性矩阵
    displacement_point = point_pos - new_com  # 计算质点相对新质心的位移
    I_point = parallel_axis_theorem(np.zeros((3, 3)), mass_point, displacement_point)

    # 合并长方体和质点的惯性矩阵
    I_total = I_box_new + I_point

    return I_total, new_com


class GO2Torque(LeggedRobot):
    cfg: GO2TorqueCfg

    def __init__(self, cfg: GO2TorqueCfg, sim_params, physics_engine, sim_device, headless):
        self.max_torque_scale = cfg.growth.max_torque_scale
        self.start_torque_scale = cfg.growth.start_torque_scale
        self.max_rear_torque_scale = cfg.growth.max_rear_torque_scale
        self.start_rear_torque_scale = cfg.growth.start_rear_torque_scale
        self.max_freq = cfg.growth.max_freq
        self.start_freq = cfg.growth.start_freq
        self.step_count = 0
        self.current_dt = 0
        self.current_freq = self.start_freq

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.motor_fatigue = torch.zeros(cfg.env.num_envs, self.num_dof, device=sim_device)
        self.torques = torch.zeros(cfg.env.num_envs, self.num_dof, device=sim_device)
        self.activation_sign = torch.zeros(cfg.env.num_envs, self.num_dof, device=sim_device)
        if self.cfg.domain_rand.loss_action_obs == False:
            self.cfg.domain_rand.loss_rate = 0

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id == 0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                              requires_grad=False)
            self.termination_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                          requires_grad=False)
            self.soft_dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device,
                                                   requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()

                self.termination_dof_pos_limits[i, 0] = self.dof_pos_limits[i, 0] - 0.05
                self.termination_dof_pos_limits[i, 1] = self.dof_pos_limits[i, 1] + 0.05

                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()

                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.soft_dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.soft_dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def check_termination(self):
        dof_pos_limits_up = self.termination_dof_pos_limits[:, 1]
        dof_pos_limits_low = self.termination_dof_pos_limits[:, 0]
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_buf |= torch.any(self.dof_pos > dof_pos_limits_up, dim=1)
        self.reset_buf |= torch.any(self.dof_pos < dof_pos_limits_low, dim=1)
        self.reset_buf |= self.projected_gravity[:, 2] > 0
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def _process_rigid_body_props(self, props, env_id):
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            com_rng_x = self.cfg.domain_rand.shifted_com_range_x
            com_rng_y = self.cfg.domain_rand.shifted_com_range_y
            com_rng_z = self.cfg.domain_rand.shifted_com_range_z
            rnd_mass = np.random.uniform(rng[0], rng[1])
            point_mass_pos = np.array([np.random.uniform(com_rng_x[0], com_rng_x[1]),
                                       np.random.uniform(com_rng_y[0], com_rng_y[1]),
                                       np.random.uniform(com_rng_z[0], com_rng_z[1])])

            # Add a point mass to the base
            props[0].mass += rnd_mass
            com_prev = np.array([props[0].com.x, props[0].com.y, props[0].com.z])
            inertia_prev = np.array([[props[0].inertia.x.x, props[0].inertia.x.y, props[0].inertia.x.z],
                                     [props[0].inertia.y.x, props[0].inertia.y.y, props[0].inertia.y.z],
                                     [props[0].inertia.z.x, props[0].inertia.z.y, props[0].inertia.z.z]])
            intertia, com = update_inertia(inertia_prev, props[0].mass, com_prev, rnd_mass, point_mass_pos)
            props[0].inertia.x += gymapi.Vec3(intertia[0, 0], intertia[0, 1], intertia[0, 2])
            props[0].inertia.y += gymapi.Vec3(intertia[1, 0], intertia[1, 1], intertia[1, 2])
            props[0].inertia.z += gymapi.Vec3(intertia[2, 0], intertia[2, 1], intertia[2, 2])
            props[0].com = gymapi.Vec3(com[0], com[1], com[2])
            for i in range(len(props)):
                props[i].mass += np.random.uniform(rng[0] / 16, rng[1] / 16)

        return props

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = (
                self.default_dof_pos * torch_rand_float(0.95, 1.05, (len(env_ids), self.num_dof), device=self.device))
        self.dof_vel[env_ids] = 0.
        self.activation_sign[env_ids] = 0.

        if self.add_noise:
            self.motor_fatigue[env_ids] = torch_rand_float(0, 0.2 * self.general_scale, (len(env_ids), self.num_dof),
                                                           device=self.device).squeeze(
                1)
        else:
            self.motor_fatigue[env_ids] = torch.zeros_like(self.motor_fatigue[env_ids])

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2),
                                                              device=self.device)  # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _update_growth_scale(self):
        self.step_count += 1
        if self.cfg.control.control_type == "T" or self.cfg.test.use_test:
            self.step_count = GO2TorqueCfgPPO().runner.num_steps_per_env * self.cfg.test.checkpoint

        # follow Gompertz curve
        self.general_scale = np.exp(-np.exp((-self.cfg.growth.k * (self.step_count - self.cfg.growth.x0))))

        self.current_freq = self.general_scale * (self.max_freq - self.start_freq) + self.start_freq
        self.current_torque_limit_scale = self.general_scale * (
                self.max_torque_scale - self.start_torque_scale) + self.start_torque_scale
        self.r_leg_scaled = self.general_scale * (
                self.max_rear_torque_scale - self.start_rear_torque_scale) + self.start_rear_torque_scale

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.actions = actions.to(self.device)
        # step physics and render each frame
        self.render()
        self.rew_buf[:] = 0.
        while self.current_dt * self.current_freq < 1:
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
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

    def _compute_torques(self, actions):
        self._update_growth_scale()
        actions_scaled = actions[:, :12] * self.cfg.control.action_scale
        self.torques_action = actions_scaled
        torques_limits = self.current_torque_limit_scale * self.torque_limits
        torques_limits[6:] = torques_limits[6:] * self.r_leg_scaled

        # muscle activation process
        if self.cfg.control.activation_process:
            current_activation_sign = torch.tanh(self.torques_action / torques_limits)
            # activation_sign = current_activation_sign
            activation_sign = (current_activation_sign - self.activation_sign) * 0.4 + self.activation_sign
        else:
            activation_sign = self.torques_action / torques_limits
        self.activation_sign = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            activation_sign, self.activation_sign)

        # hill model
        if self.cfg.control.hill_model:
            self.torques = self.activation_sign * torques_limits * (
                    1 - torch.sign(self.activation_sign) * self.dof_vel / self.dof_vel_limits)
        else:
            self.torques = self.activation_sign * torques_limits

        # fatigue
        if self.cfg.control.motor_fatigue:
            self.motor_fatigue += torch.abs(self.torques) * self.dt
            self.motor_fatigue *= 0.9
        else:
            self.motor_fatigue = torch.zeros_like(self.motor_fatigue)

        return self.torques

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:21] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[21:33] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[33:38] = 0.  # commands
        noise_vec[38:49] = 0.  # actions
        noise_vec[50:62] = noise_scales.fatigue * noise_level / 10
        if self.cfg.terrain.measure_heights:
            noise_vec[64:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity.
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy * self.general_scale
        self.root_states[:, 7:10] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 3),
                                                     device=self.device)  # lin vel x/y

        max_ang_vel = self.cfg.domain_rand.max_push_vel_ang * self.general_scale
        self.root_states[:, 10:13] = torch_rand_float(-max_ang_vel, max_ang_vel, (self.num_envs, 3),
                                                      device=self.device)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def compute_observations(self):
        base_lin_vel = self.base_lin_vel
        # base_lin_vel[:, 2] = 0
        obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.commands[:, :3] * self.commands_scale[:3],
                             self.commands[:, 4:5] * self.commands_scale[4:5],
                             self.commands[:, 5:6] * self.commands_scale[5:6],
                             self.torques,
                             self.motor_fatigue
                             ), dim=-1)
        self.obs_buf = torch.cat((base_lin_vel * self.obs_scales.lin_vel,
                             self.base_ang_vel * self.obs_scales.ang_vel,
                             self.projected_gravity,
                             (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                             self.dof_vel * self.obs_scales.dof_vel,
                             self.commands[:, :3] * self.commands_scale[:3],
                             self.commands[:, 4:5] * self.commands_scale[4:5],
                             self.commands[:, 5:6] * self.commands_scale[5:6],
                             self.torques,
                             self.motor_fatigue
                             ), dim=-1)
        # print("Observation Buffer Shape:", self.obs_buf.shape)
        # print("Body Height Command in Observation:", self.commands[:, 5])

        # add noise if needed
        if self.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * self.noise_scale_vec

        self.obs_buf = torch.where(
            torch.rand(self.num_envs, device=self.device).unsqueeze(1) > self.cfg.domain_rand.loss_rate,
            obs_buf, self.obs_buf)

    def _init_buffers(self):
        super()._init_buffers()
        self.prev_dof_pos = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.hip_indices = []
        self.thigh_indices = []
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if 'hip_joint' in name:
                self.hip_indices.append(i)
            if 'thigh_joint' in name:
                self.thigh_indices.append(i)
        self.torque_limits = torch.ones(self.num_dof, device=self.device) * 23.5
        rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self.general_scale = 0
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float,
                                    device=self.device, requires_grad=False)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, 1.0, self.obs_scales.pitch,self.obs_scales.body_height],
                                           device=self.device, requires_grad=False, )  # TODO change this
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        if self.cfg.test.use_test and self.cfg.control.control_type == "T":
            self.commands[:] = self.cfg.test.vel
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                                             self.command_ranges["ang_vel_yaw"][0] * self.general_scale,
                                             self.command_ranges["ang_vel_yaw"][1] * self.general_scale)

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        x_cmd_sum = self.command_ranges["lin_vel_x"][1] + self.command_ranges["lin_vel_x"][0]
        x_cmd_diff = self.command_ranges["lin_vel_x"][1] - self.command_ranges["lin_vel_x"][0]
        self.commands[env_ids, 0] = torch_rand_float(
            max(x_cmd_sum * 0.5 - x_cmd_diff * self.general_scale, self.command_ranges["lin_vel_x"][0]),
            min(x_cmd_sum * 0.5 + x_cmd_diff * self.general_scale, self.command_ranges["lin_vel_x"][1]),
            (len(env_ids), 1),
            device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0] * self.general_scale,
                                                     self.command_ranges["lin_vel_y"][1] * self.general_scale,
                                                     (len(env_ids), 1),
                                                     device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],
                                                         self.command_ranges["heading"][1],
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0] * self.general_scale,
                                                         self.command_ranges["ang_vel_yaw"][1] * self.general_scale,
                                                         (len(env_ids), 1),
                                                         device=self.device).squeeze(1)
        if self.cfg.commands.pitch_command:
            self.commands[env_ids, 4] = torch_rand_float(
                self.command_ranges["pitch"][0],
                self.command_ranges["pitch"][1],
                (len(env_ids), 1),
                device=self.device).squeeze(1)

        if self.cfg.commands.body_height_command:
            self.commands[env_ids, 5] = torch_rand_float(
                self.command_ranges["body_height"][0],
                self.command_ranges["body_height"][1],
                (len(env_ids), 1),
                device=self.device).squeeze(1)

    ##############################################################################################################

    def _reward_pitch(self):
        """奖励机器人接近目标俯仰角"""
        if not self.cfg.commands.pitch_command:
            return torch.zeros(self.num_envs, device=self.device)
        reward = torch.exp(
            -torch.abs(-self.projected_gravity[:, 0] - self.commands[:, 4]) / self.cfg.rewards.pitch_tracking_sigma
        )* self.general_scale
        print(f"Actual pitch: {-self.projected_gravity[:, 0]}")
        # print(f"Target pitch: {self.commands[:, 4]}")
        return reward

    def _reward_soft_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.soft_dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.soft_dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_forward(self):
        # Reward for moving forward
        return torch.exp(
            -torch.abs(self.base_lin_vel[:, 0] - (
                    self.cfg.commands.ranges.lin_vel_x[1] + self.cfg.commands.ranges.lin_vel_x[0]) / 2 * max((
                    1 - self.general_scale * 2), 0) - self.commands[:, 0] * min(self.general_scale * 2, 1)) /
            self.cfg.rewards.tracking_sigma)
            # max((0.5 - self.general_scale), self.cfg.rewards.tracking_sigma))

    def _reward_head_height(self):
        """奖励机器人保持合适的机身高度，并在pitch_command关闭时执行head_up奖励"""

        # 计算当前基座高度（机器人 body 相对于地面的高度）
        current_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        # 如果 body height command 开启，则计算基于 commands[:, 5] 的高度奖励
        if self.cfg.commands.body_height_command:
            reward = torch.exp(-torch.abs(
                current_height - self.commands[:, 5]) / self.cfg.rewards.height_tracking_sigma) * self.general_scale
            print(f"Actual height: {current_height}")
            # print(f"Target height: {self.commands[:, 5]}")

        else:
            # 否则，计算基于固定 target height 的奖励
            base_height = current_height.clip(max=self.cfg.rewards.base_height_target)
            reward = base_height * (1 + self.general_scale)
            # print(f"Actual height: {base_height}")
            # print(f"Target height: {self.cfg.rewards.base_height_target}")

        # 计算 head_up 额外奖励（仅当 pitch command 关闭时才执行）
        if not self.cfg.commands.pitch_command:
            head_up = -(self.projected_gravity[:, 0].clip(min=min(0, -0.2 * (1.5 - self.general_scale * 2))))
            reward += head_up  # 叠加 head up 奖励

        return reward

    def _reward_moving_y(self):
        return torch.exp(-torch.abs(
            self.base_lin_vel[:, 1] - self.commands[:, 1]) / self.cfg.rewards.tracking_sigma) * self.general_scale

    def _reward_moving_yaw(self):
        return torch.exp(-torch.abs(
            self.base_ang_vel[:, 2] - self.commands[:, 2]) / self.cfg.rewards.tracking_sigma) * self.general_scale

    def _reward_motor_fatigue(self):
        return torch.sum(self.motor_fatigue * torch.abs(self.torques_action), dim=1)

    def _reward_roll(self):
        return torch.abs(self.projected_gravity[:, 1])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])
