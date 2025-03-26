
from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.draw_arrow import WireframeArrowGeometry

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .go2_arx_config import Go2ArxRoughCfg
from  legged_gym.envs.base.legged_robot import LeggedRobot

from .go2_arx_robot import Go2ArxRobot
from .go2_arx_config_high_level import Go2ArxHLRoughCfg
from .go2_arx_config import Go2ArxRoughCfg
#from loco_manipulation_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

class Go2ArxHLRobot(Go2ArxRobot):
    # cfg: Go2ArxHLRoughCfg
    def __init__(self, cfg: Go2ArxHLRoughCfg, sim_params, physics_engine, sim_device, headless):
        self.cfg_HL = cfg #high level config
        self.num_actions_HL = self.cfg_HL.env.num_actions
        self.num_obs_HL = self.cfg_HL.env.num_observations
        self.cfg = Go2ArxRoughCfg() #low level config
        self.num_obs = self.cfg.env.num_observations
        self.num_actions = self.cfg.env.num_actions
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self._load_low_level_policy()
        self.post_physics_step()

    def _load_low_level_policy(self):
        policy_path = os.path.join(LEGGED_GYM_ROOT_DIR, self.cfg_HL.policy.policy_path,)
        self.policy = torch.jit.load(os.path.join(str(policy_path), self.cfg_HL.policy.policy_name)).to(self.device)
        self.policy.eval()

    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_HL = torch.zeros(self.num_envs, self.num_actions_HL, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions_HL = torch.zeros(self.num_envs, self.num_actions_HL, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg_HL.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.init_ee_goal_variale()


        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0

        # random motor lenth
        str_rng = self.cfg.domain_rand.motor_strength_range
        self.motor_strength = (str_rng[1] - str_rng[0]) * torch.rand(2, self.num_envs, self.num_dof, dtype=torch.float, device=self.device, requires_grad=False) + str_rng[0]
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

    def step(self, actions):# TODO how does action turns into robot's movement
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # clip_actions = self.cfg.normalization.clip_actions
        # self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.process_actions(actions) #TODO
        self.compute_observations()
        self.actions = self.policy(self.obs_buf.detach()).detach()
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques,self.arm_pos = self._compute_torques(self.actions)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_pos))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.post_physics_step()#TODO

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def compute_observations_HL(self):#TODO
        """ Computes observations
        """
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.wheel_indices] = 0
        self.dof_pos[:,self.wheel_indices] = 0
        self.obs_buf_HL = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :4] * self.commands_scale,
                                    self.dof_err * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self._local_gripper_pos*self.obs_scales.gripper_track,
                                    # torch.flatten(self.future_ee_goal_sphere, start_dim=1)*self.obs_scales.gripper_track,
                                    quat_rotate_inverse(self.base_quat, self.commands[:, -3:])*self.obs_scales.commands*self.obs_scales.gripper_track,
                                    self.actions_HL,
                                    self.actions
                                    ),dim=-1)
        # self.obs_buf = torch.cat((  self.actions,
        #                             self.commands[:, :4] * self.commands_scale,
        #                             quat_rotate_inverse(self.base_quat, self.commands[:, -3:])*self.obs_scales.commands, 
        #                             ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            if not self.cfg.env.symmetric:
                 self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _resample_commands(self, env_ids):
        omega = 0.2 * torch.ones(len(env_ids), device=self.device)
        r = 0.05 * torch.ones(len(env_ids), device=self.device)
        counter = self.episode_length_buf[env_ids]
        self.commands[env_ids, 0] = omega * counter * self.dt# r * torch.cos(omega * self.common_step_counter * self.dt) # p_x
        self.commands[env_ids, 1] = torch.zeros_like(self.commands[env_ids, 1], device=self.device) #r * torch.sin(omega * self.common_step_counter * self.dt) # p_y
        self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device) # p_z
        self.commands[env_ids, :] += self.env_origins[env_ids, :]

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        target_ee = self.commands[:, :]

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        upper_arm_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))

        # cone = WireframeArrowGeometry()
        # gymutil.draw_lines(cone, self.gym, self.viewer, self.envs[0], gymapi.Transform(gymapi.Vec3(5, 5, 5), r=None))
        

        ee_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self._local_gripper_pos)
        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(target_ee[i, 0], target_ee[i, 1], target_ee[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 

    def post_physics_step(self):
        super().post_physics_step()
        self.last_actions_HL = self.actions_HL

    def compute_observations(self): # compute observations for low level policy
        """ Computes observations
        """
        self.dof_err = self.dof_pos - self.default_dof_pos
        self.dof_err[:,self.wheel_indices] = 0
        self.dof_pos[:,self.wheel_indices] = 0
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    self.commands[:, :3] * self.commands_scale,
                                    self.dof_err * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self._local_gripper_pos*self.obs_scales.gripper_track,
                                    self.curr_ee_goal_cart*self.obs_scales.gripper_track,
                                    # torch.flatten(self.future_ee_goal_sphere, start_dim=1)*self.obs_scales.gripper_track,
                                    (self._local_gripper_pos-self.curr_ee_goal_cart)*self.obs_scales.gripper_track,
                                    self.actions
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            if not self.cfg.env.symmetric:
                 self.privileged_obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # if self.add_noise:
        #     self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec #TODO

    def process_actions(self, actions):
        self.actions_HL = actions.to(self.device)
        self.action_HL_upper = torch.tensor([1, 1, 1.5, 1, 1, 1, 1],device=self.device).repeat((self.num_envs, 1))
        self.action_HL_lower = torch.tensor([-1, -1, 0.2, 0, -1, -1, -1],device=self.device).repeat((self.num_envs, 1))
        self.actions_HL = torch.clamp(self.actions, self.action_HL_lower, self.action_HL_upper)

    # #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        dof_err = self.dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        dof_err[:,self.arm_indices] = 0
        return torch.sum(torch.abs(dof_err), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
    
    def _reward_joint_pos_rate(self):
        # Penalize motion at zero commands
        dof_err = self.last_dof_pos - self.default_dof_pos
        dof_err[:,self.wheel_indices] = 0
        return torch.sum(torch.square(dof_err), dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
    def _reward_object_distance(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos-quat_rotate_inverse(self.base_quat, self.commands[:, -3:])), dim=1)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return torch.exp(-dis_err/0.1)
    
    def _reward_object_distance_l2(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos-quat_rotate_inverse(self.base_quat, self.commands[:, -3:])), dim=1)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return dis_err