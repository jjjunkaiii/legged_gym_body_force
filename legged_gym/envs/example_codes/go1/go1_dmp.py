import os

import torch
from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.go1.go1 import Go1
from legged_gym.utils.math import quat_apply_yaw
from .go1_dmp_config import Go1DmpCfg
import random


class Go1DMP(Go1):

    def __init__(self, cfg: Go1DmpCfg, sim_params, physics_engine, sim_device, headless):
        self.cpg_cfg = cfg.cpg_policy
        self.student_cfg = cfg.task_for_student
        self.resample_count = 0
        self.draw_count = 0
        self.traj_record = []

        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.num_envs_indexes = list(range(0, self.num_envs))
        self.randomized_lag = [random.randint(0, self.cfg.domain_rand.lag_timesteps - 1) for i in range(self.num_envs)]
        self.lag_buffer = torch.zeros(self.num_envs, self.cfg.domain_rand.lag_timesteps, self.num_actions,
                                      device=self.device, requires_grad=False)
        self.heights = torch.zeros(self.num_envs, 160, dtype=torch.float, device=self.device, requires_grad=False)
        self.cfg = cfg
        if self.student_cfg.use_student and self.student_cfg.rand_map_update:
            self.student_cfg.student_obs_num = 76 + self.num_actions + 160
            self.student_obs_buf = torch.zeros(self.num_envs, self.student_cfg.student_obs_num, dtype=torch.float,
                                               device=self.device, requires_grad=False)
            if self.student_cfg.rand_map_update:
                self.map_update_count = torch.zeros(self.num_envs, 1, dtype=torch.int, device=self.device,
                                                    requires_grad=False)
                self.map_update = torch.randint(self.student_cfg.map_update_range[0],
                                                self.student_cfg.map_update_range[1], (self.num_envs, 1),
                                                device=self.device)
            if self.student_cfg.add_delay:
                self.obs_delay = [torch.zeros(self.num_envs, 40, device=self.device, requires_grad=False)]
            if self.student_cfg.add_noise:
                self.student_noise_scale_vec = self._get_noise_scale_vec_student()
        self._load_cpg_policy()
        print(self.cpg_policy)

    def step(self, actions):
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        obs_cpg = self.obs_buf[:, :self.cpg_obs_num].detach()
        actions_cpg = self.cpg_policy(obs_cpg).detach()
        self.actions_cpg = torch.clip(actions_cpg, -self.cpg_action_clip, self.cpg_action_clip)
        self.actions_scaled_cpg = actions_cpg * self.cpg_action_scale

        self.render()
        for _ in range(self.decimation):
            self.torques = self._compute_torques().view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _get_heights_1(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points_1, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points_1),
                                    self.height_points_1[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points_1), self.height_points_1) + (
                self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points / self.terrain.cfg.horizontal_scale).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px + 1, py]
        heights3 = self.height_samples[px, py + 1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

    # --------callback-------------

    def compute_observations(self):
        self.obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos + (
                    2 * torch.rand_like(self.dof_pos) - 1) * 0.4 * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel + (
                    2 * torch.rand_like(self.dof_vel) - 1) * 6.0 * self.obs_scales.dof_vel,
            self.contact_forces[:, self.feet_indices, 2] > 1.,
            (self._cpg.X[:, 0, :] - ((self._cpg.mu_up[0] + self._cpg.mu_low[
                0]) / 2)) * self.obs_scales.dof_pos,
            (self._cpg.X[:, 1, :] - np.pi) * 1 / np.pi,
            (self._cpg.X[:, 2, :] - np.pi) * 1 / np.pi,
            self._cpg.X_dot[:, 0, :] * 1 / 30,
            (self._cpg.X_dot[:, 1, :] - 15) * 1 / 30,
            (self._cpg.X_dot[:, 2, :] - 15) * 1 / 30,
            self.actions_cpg,
            self.actions,
            # some extra observations
            self.payloads.unsqueeze(1),
            self.com_displacements,
            self.friction_coeffs,
            self.restitutions,
        ), dim=-1)

        if self.student_cfg.use_student:
            self.student_obs_buf[:, :76 + self.num_actions] = self.obs_buf[:, :76 + self.num_actions]
            if self.student_cfg.add_delay:
                self.obs_delay.append(self.obs_buf[:, :40])
                if len(self.obs_delay) > self.student_cfg.delay_count:
                    obs_buf = self.obs_delay.pop(0)
                    self.student_obs_buf[:, :9] = obs_buf[:, :9]


        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1,
                                 1.)
            heights = heights * self.obs_scales.height_measurements
            if self.student_cfg.use_student and self.student_cfg.rand_map_update:
                self.map_update_count += 1
                remainder = torch.remainder(self.map_update_count, self.map_update)
                env_idxs = torch.where(remainder == 0)[0]
                self.student_obs_buf[env_idxs, -160:] = heights[env_idxs]
                self.map_update_count[env_idxs] = 0
                self.map_update[env_idxs, :] = torch.randint(self.student_cfg.map_update_range[0],
                                                             self.student_cfg.map_update_range[1],
                                                             (len(env_idxs), 1),
                                                             device=self.device)

            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)

        if self.student_cfg.use_student and self.student_cfg.add_noise:
            self.student_obs_buf += (2 * torch.rand_like(self.student_obs_buf) - 1) * self.student_noise_scale_vec

        self.obs_buf = torch.cat((self.obs_buf, self.student_obs_buf), dim=-1)

    def _compute_torques(self):
        self.actions_scaled = self.actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if "CPG" in control_type:
            xs, ys, zs = self._cpg.get_CPG_RL_actions(self.actions_scaled_cpg, self.frequency_high,
                                                      self.frequency_low)
            des_joint_pos = torch.zeros_like(self.torques, device=self.device)
            sideSign = np.array([-1, 1, -1, 1])
            foot_y = torch.ones(self.num_envs, device=self.device,
                                requires_grad=False) * self.cfg.asset.hip_link_length_a1
            LEG_INDICES = np.array([1, 0, 3, 2])
            for ig_idx, i in enumerate(LEG_INDICES):
                if self.cfg.domain_rand.randomize_lag_timesteps:
                    self.lag_buffer = torch.cat([self.lag_buffer[:, 1:, :].clone(), self.actions_scaled.unsqueeze(1).clone()], dim=1)
                    x = self.lag_buffer[self.num_envs_indexes, self.randomized_lag, 4 + i] + xs[:, i]
                    z = self.lag_buffer[self.num_envs_indexes, self.randomized_lag, i] + zs[:, i]
                else:
                    x = xs[:, i] + self.actions_scaled[:, 4 + i]
                    z = zs[:, i] + self.actions_scaled[:, i]

                y = sideSign[i] * foot_y + ys[:, i] + self.actions_scaled[:, i + 8]
                robot_length = self.cfg.asset
                des_joint_pos[:, 3 * ig_idx:3 * ig_idx + 3] = self._cpg.compute_inverse_kinematics(robot_length, i, x,
                                                                                                   y, z)
            self.dof_des_pos = des_joint_pos
            if self.cfg.control.use_actuator_net:
                self.joint_vel = self.dof_vel
                torques = self.actuator_network(self.dof_des_pos, self.joint_pos_err_last,
                                                self.joint_pos_err_last_last,
                                                self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
                self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
                self.joint_pos_err_last = torch.clone(self.dof_des_pos)
                self.joint_vel_last_last = torch.clone(self.joint_vel_last)
                self.joint_vel_last = torch.clone(self.joint_vel)
            else:
                torques = self.p_gains * self.Kp_factors * (
                        self.dof_des_pos - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
            return torch.clip(torques, -self.torque_limits, self.torque_limits)
        else:
            raise NotImplementedError(f"Control type {control_type} not implemented")

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.previous_ang_vel[env_ids] = torch.zeros(len(env_ids), 5, device=self.device, requires_grad=False)
        if self.student_cfg.use_student and self.student_cfg.rand_map_update:
            self.map_update_count[env_ids] = 0
            self.map_update[env_ids, :] = torch.randint(self.student_cfg.map_update_range[0],
                                                        self.student_cfg.map_update_range[1], (len(env_ids), 1),
                                                        device=self.device)
            if self.student_cfg.add_delay:
                self.obs_delay = [torch.zeros(self.num_envs, 40, device=self.device, requires_grad=False)]

            self.lag_buffer[env_ids, :, :] = 0

    def _post_physics_step_callback(self):
        if self.cfg.terrain.measure_heights:
            self.measured_heights_1 = self._get_heights_1()
        super()._post_physics_step_callback()

    # --------init function--------
    def _load_cpg_policy(self):
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, self.cpg_cfg.policy_path, 'exported', 'policies')
        self.cpg_policy = torch.jit.load(os.path.join(str(policy_path), 'policy_1.pt')).to(self.device)
        self.cpg_policy.eval()

        self.cpg_obs_num = self.cpg_cfg.cpg_obs_num
        self.cpg_action_scale = self.cpg_cfg.normalization.action_scale
        self.cpg_action_clip = self.cpg_cfg.normalization.clip_actions
        self.cpg_action_num = self.cpg_cfg.cpg_action_num
        self.actions_cpg = torch.zeros(self.num_envs, self.cpg_action_num, dtype=torch.float, device=self.device,
                                       requires_grad=False)
        self.post_physics_step()

    def _get_noise_scale_vec(self):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:76] = 0.  # previous actions,contact bool, CPG, CPG actions
        noise_vec[76:76 + self.num_actions] = 0.  # actions
        noise_vec[76 + self.num_actions: 76 + self.num_actions + self.cfg.env.num_extra_observations] = 0.  # extra obs
        if self.cfg.terrain.measure_heights:
            noise_vec[
            76 + self.num_actions + self.cfg.env.num_extra_observations:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _get_noise_scale_vec_student(self):
        noise_vec = torch.zeros_like(self.student_obs_buf[0])
        noise_scales = self.student_cfg.noise.noise_scales
        noise_level = self.student_cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:12] = 0.  # commands
        noise_vec[12:24] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[24:36] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[36:76] = 0.  # previous actions,contact bool, CPG, CPG actions
        noise_vec[76:76 + self.num_actions] = 0.  # actions
        if self.cfg.terrain.measure_heights:
            noise_vec[
            76 + self.num_actions:] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements
        return noise_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.previous_ang_vel = torch.zeros(self.num_envs, 5, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points_1 = self._init_height_points_1()
        self.measured_heights_1 = 0

    def _init_height_points_1(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """

        y_1 = torch.tensor(self.cfg.terrain.measured_points_y_1, device=self.device, requires_grad=False)
        x_1 = torch.tensor(self.cfg.terrain.measured_points_x_1, device=self.device, requires_grad=False)
        grid_x_1, grid_y_1 = torch.meshgrid(x_1, y_1)

        self.num_height_points_1 = grid_x_1.numel()
        points_1 = torch.zeros(self.num_envs, self.num_height_points_1, 3, device=self.device, requires_grad=False)
        points_1[:, :, 0] = grid_x_1.flatten()
        points_1[:, :, 1] = grid_y_1.flatten()
        return points_1

    # -------reward function--------
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_energy(self):
        return torch.sum(torch.abs(self.torques * (self.dof_vel - self.last_dof_vel)), dim=1)

    def _reward_fly(self):
        # Penalize flying
        fly = torch.ones(self.num_envs, device=self.device, requires_grad=False)
        fly = (torch.sum(self.contact_forces[:, self.feet_indices, 2] < 1., dim=1) >= 3) * fly

        return fly

    def _reward_feet_air_time(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt

        # Apply punishment for too long air time
        punishment_factor = torch.exp(self.feet_air_time - 2.0)  # exponential punishment factor
        punishment = torch.sum(punishment_factor * (self.feet_air_time > 2.0),
                               dim=1)  # apply punishment for air time > 1.0
        rew_airTime = -punishment

        # Reset air time for feet that are in contact with the ground
        self.feet_air_time *= ~contact_filt

        return rew_airTime

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(
            self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_action_xy(self):
        # Penalize large actions
        return torch.sum(torch.square(self.actions[:, 4:]), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error * (5 * torch.exp(-lin_vel_error) + 1))

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights_1, dim=1)
        return torch.clip(torch.square(base_height - self.cfg.rewards.base_height_target), -1, 1)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error * (5 * torch.exp(-ang_vel_error) + 1))

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
