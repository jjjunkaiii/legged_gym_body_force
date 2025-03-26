from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.math import quat_apply_yaw
import torch

class Go1Legged(LeggedRobot):
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.steps_until_update = torch.ones(self.num_envs, device=self.device, requires_grad=False)
        self.heights = torch.zeros(self.num_envs, 132, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_torques = torch.zeros(self.num_envs, self.num_dofs, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points_1 = self._init_height_points_1()
        self.measured_heights_1 = 0.0

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.last_torques[env_ids] = 0.0

    def _get_noise_scale_vec(self, cfg):
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
        noise_vec[36:48] = 0.  # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[48:235] = noise_scales.height_measurements * noise_level * self.obs_scales.height_measurements

        self.noise_scale_vec_p_gains = torch.zeros(self.num_dofs, device=self.device, requires_grad=False)
        self.noise_scale_vec_d_gains = torch.zeros(self.num_dofs, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_PD:
            self.noise_scale_vec_p_gains[:] = noise_scales.p_gains * noise_level
            self.noise_scale_vec_d_gains[:] = noise_scales.d_gains * noise_level
        return noise_vec

    def _compute_torques(self, actions):
        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        if control_type == "P":
            p_gains = (2 * torch.rand_like(self.p_gains) - 1) * self.noise_scale_vec_p_gains + self.p_gains
            d_gains = (2 * torch.rand_like(self.d_gains) - 1) * self.noise_scale_vec_d_gains + self.d_gains
            torques = p_gains * (actions_scaled + self.default_dof_pos - self.dof_pos) - d_gains * self.dof_vel
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _init_height_points_1(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """

        y_1 = torch.tensor(self.cfg.terrain.measured_points_y_base, device=self.device, requires_grad=False)
        x_1 = torch.tensor(self.cfg.terrain.measured_points_x_base, device=self.device, requires_grad=False)
        grid_x_1, grid_y_1 = torch.meshgrid(x_1, y_1)

        self.num_height_points_1 = grid_x_1.numel()
        points_1 = torch.zeros(self.num_envs, self.num_height_points_1, 3, device=self.device, requires_grad=False)
        points_1[:, :, 0] = grid_x_1.flatten()
        points_1[:, :, 1] = grid_y_1.flatten()
        return points_1

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

    def _post_physics_step_callback(self):
        if self.cfg.terrain.measure_heights:
            self.measured_heights_1 = self._get_heights_1()
        super()._post_physics_step_callback()

    def post_physics_step(self):
        super().post_physics_step()
        self.last_torques[:] = self.torques[:]

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_energy(self):
        # Penalize energy
        return torch.sum(torch.abs(self.torques * self.dof_vel), dim=1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :],
                                     dim=-1) - self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _reward_feet_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > 5 * torch.abs(
            self.contact_forces[:, self.feet_indices, 2]), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1. * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1),
                         dim=1)

    def _reward_delta_torques(self):
        return torch.sum(torch.square(self.torques - self.last_torques), dim=1)

    def _reward_action(self):
        # Penalize large actions
        return torch.sum(torch.square(self.actions), dim=1)

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)
