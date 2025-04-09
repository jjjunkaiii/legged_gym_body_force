
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
#from loco_manipulation_gym.utils.isaacgym_utils import get_euler_xyz as get_euler_xyz_in_tensor

class Go2ArxRobot(LeggedRobot):
    cfg: Go2ArxRoughCfg

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # get actuator mode
        # props = self.gym.get_actor_dof_properties(self.envs[0], self.actor_handles[0],)
        # print("###props:",props)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques,self.arm_pos = self._compute_torques(self.actions)
            # self.torques[:,-6:] = 0
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.arm_pos))
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

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
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

        self._post_physics_step_callback()
        self.update_curr_ee_goal()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_pos[:] = self.dof_pos[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer :
            self._draw_ee_goal_track()
            self._draw_debug_vis()

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)
        self.update_curr_ee_goal()
        
        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    
    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
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
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_base_mass:
            rng = self.cfg.domain_rand.added_mass_range
            props[0].mass += np.random.uniform(rng[0], rng[1])
        if self.cfg.domain_rand.randomize_base_com:
            rng_com = self.cfg.domain_rand.added_com_range
            rand_com = np.random.uniform(rng_com[0], rng_com[1], size=(3, ))
            props[0].com += gymapi.Vec3(*rand_com)
        return props
    
    def orientation_error(self,desired, current):
        cc = quat_conjugate(current)
        q_r = quat_mul(desired, cc)
        return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)
       
    def control_ik(self,local_ee_pose,local_goal_pose,local_j_eef): #TODO: task space potential field
        pos_err = local_goal_pose[:,0:3] - local_ee_pose[:,0:3]
        orn = torch.tensor([0,0,0,1], device=self.device).repeat(self.num_envs, 1)
        orn_err = self.orientation_error(orn, orn)
        dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)
        # solve damped least squares
        j_eef_T = torch.transpose(local_j_eef, 1, 2)
        lmbda = torch.eye(6, device=self.device) * (0.05 ** 2)
        u = (j_eef_T @ torch.inverse(local_j_eef @ j_eef_T + lmbda) @ dpose).view(self.num_envs, 6)
        return u

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """

        # wheel control 
        self.dof_err =  self.dof_pos -self.default_dof_pos
        self.dof_err[:,self.wheel_indices] = 0
        #arm ik 
        ik_u = self.control_ik(self._local_gripper_pos,self.curr_ee_goal,self.j_eef)
        # self.arm_u[:,self.arm_indices] = self.dof_pos[:,self.arm_indices]  + actions[:,self.arm_indices] + ik_u
        self.arm_u[:,self.arm_indices] = self.dof_pos[:,self.arm_indices]+ actions[:,self.arm_indices] + ik_u

        actions_scaled = actions * self.cfg.control.action_scale
        control_type = self.cfg.control.control_type
        modify_dof_vel = self.dof_vel.clone().detach()
        modify_dof_vel[:,self.arm_indices] = 0
        if control_type=="P":
            if not self.cfg.domain_rand.randomize_motor:  # TODO add strength to gain directly
                torques = self.p_gains*(actions_scaled - self.dof_err) - self.d_gains*modify_dof_vel
            else:
                torques = self.motor_strength[0] * self.p_gains*(actions_scaled - self.dof_err) - self.motor_strength[1] * self.d_gains*modify_dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - modify_dof_vel) - self.d_gains*(modify_dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits),self.arm_u

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
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_dof_pos = torch.zeros_like(self.dof_pos)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
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


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.robot_asset = robot_asset
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        arm_names =[]
        for name in self.cfg.asset.arm_joint_name:
            arm_names.extend([s for s in self.dof_names if name in s])
        wheel_names =[]
        for name in self.cfg.asset.wheel_joint_name:
            wheel_names.extend([s for s in self.dof_names if name in s])

        leg_names =[]
        for name in self.cfg.asset.leg_joint_name:
            leg_names.extend([s for s in self.dof_names if name in s])

        print("###self.rigid_body names:",body_names)
        print("###self.dof names:",self.dof_names)
        print("###penalized_contact_names:",penalized_contact_names)
        print("###termination_contact_names:",termination_contact_names)
        print("###feet_names:",feet_names)
        print("###wheels name:",wheel_names)
        print("###arm_names:",arm_names)
        print("###leg_names:",leg_names)
        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        for envs_idx in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[envs_idx].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, envs_idx)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, envs_idx, self.cfg.asset.self_collisions, 0)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

            if envs_idx==0:
                self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(feet_names)):
                    self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

                self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(penalized_contact_names)):
                    self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

                self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(termination_contact_names)):
                    self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

                self.wheel_indices = torch.zeros(len(wheel_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(wheel_names)):
                    self.wheel_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], wheel_names[i])
                
                self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(arm_names)):
                    self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])

                self.leg_joint_indices = torch.zeros(len(leg_names), dtype=torch.long, device=self.device, requires_grad=False)
                for i in range(len(leg_names)):
                    self.leg_joint_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], leg_names[i])

                
                print("###self.wheel_indices:",self.wheel_indices)
                print("###self.arm_indices:",self.arm_indices)
                print("###self.leg_indices:",self.leg_joint_indices)

            dof_props = self._process_dof_props(dof_props_asset, envs_idx)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, envs_idx)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)

    # def _process_dof_props(self, props, env_id):
    #     super()._process_dof_props(props, env_id)

    #     for i in range(len(props)):
    #         if i < 12:
    #             # Leg joints: torque control
    #             props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
    #             props["stiffness"][i] = 0.0
    #             props["damping"][i] = 0.0
    #         else:
    #             # Arm joints: position control
    #             props["driveMode"][i] = gymapi.DOF_MODE_POS
    #             props["stiffness"][i] = 0.5# 50.0  # or higher if needed
    #             props["damping"][i] = 0.01#1.0   
    #     return props

# ee goal function
    #----------------------------------------
    def init_ee_goal_variale(self):

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        actor_jacobian = self.gym.acquire_jacobian_tensor(self.sim, self.cfg.asset.name)

        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gripperMover_handles = self.gym.find_asset_rigid_body_index(self.robot_asset, self.cfg.asset.arm_gripper_name)
        robot_link_dict = self.gym.get_asset_rigid_body_dict(self.robot_asset)

        self.hand_index = robot_link_dict[self.cfg.asset.arm_gripper_name]
        # create some wrapper tensors for different slices
        self.whole_body_jacobian = gymtorch.wrap_tensor(actor_jacobian)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)

        self.j_eef = self.whole_body_jacobian[:,self.hand_index, :6, self.arm_indices+6]

        #
        self.ee_pos = self.rigid_body_states[:, self.hand_index, :3]
        self.ee_orn = self.rigid_body_states[:, self.hand_index, 3:7]
        self.ee_vel = self.rigid_body_states[:, self.hand_index, 7:]
        self.ee_j_eef = self.whole_body_jacobian[:, self.hand_index, :6, self.arm_indices+6]

        arm_names =[]
        for name in self.cfg.asset.arm_joint_name:
            arm_names.extend([s for s in self.dof_names if name in s])

        self.arm_indices = torch.zeros(len(arm_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(arm_names)):
            self.arm_indices[i] = self.gym.find_actor_dof_handle(self.envs[0], self.actor_handles[0], arm_names[i])
        self.arm_u = torch.zeros((self.num_envs,self.num_dofs),dtype=torch.float,device=self.device)




        print("###arm_names:",arm_names)
        print("###arm_indices:",self.arm_indices)
        print("###hand_index:",self.hand_index)
        print("###gripperMover_handles:",self.gripperMover_handles)

        #update variable
        self.goal_timer = torch.zeros(self.num_envs, device=self.device)
        self.traj_timesteps = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.traj_total_timesteps = self.traj_timesteps + torch_rand_float(self.cfg.goal_ee.hold_time[0], self.cfg.goal_ee.hold_time[1], (self.num_envs, 1), device=self.device).squeeze() / self.dt
        self.ee_start_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_delta_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.ee_goal_orn_euler = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_cart = torch.zeros(self.num_envs, 3, device=self.device)
        self.curr_ee_goal_sphere = torch.zeros(self.num_envs, 3, device=self.device)
        self.collision_lower_limits = torch.tensor(self.cfg.goal_ee.collision_lower_limits, device=self.device, dtype=torch.float)
        self.collision_upper_limits = torch.tensor(self.cfg.goal_ee.collision_upper_limits, device=self.device, dtype=torch.float)
        self.underground_limit = self.cfg.goal_ee.underground_limit
        self.num_collision_check_samples = self.cfg.goal_ee.num_collision_check_samples
        self.collision_check_t = torch.linspace(0, 1, self.num_collision_check_samples, device=self.device)[None, None, :]
        self.goal_ee_ranges = class_to_dict(self.cfg.goal_ee.ranges)
        self.init_goal_ee_l_ranges = self.goal_ee_l_ranges = np.array(self.goal_ee_ranges['init_pos_l'])
        self.init_goal_ee_p_ranges = self.goal_ee_p_ranges = np.array(self.goal_ee_ranges['init_pos_p'])
        self.init_goal_ee_y_ranges = self.goal_ee_y_ranges = np.array(self.goal_ee_ranges['init_pos_y'])
        self.goal_ee_delta_orn_ranges = torch.tensor(self.goal_ee_ranges['final_delta_orn'])
        self._local_cube_object_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)
        self._cube_object_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)

        # future ee goals
        self.future_ee_goal_steps = self.cfg.goal_ee.future_ee_goal_steps
        self.future_ee_goal_dt = self.cfg.goal_ee.future_ee_goal_dt
        # self.future_ee_goal_time = torch.tensor(self.future_ee_goal_dt * torch.arange(self.future_ee_goal_steps), device=self.device)
        # assign a time vector for each env
        self.future_ee_goal_time = torch.zeros((self.num_envs, self.future_ee_goal_steps), dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            self.future_ee_goal_time[i] = self.future_ee_goal_dt * torch.arange(self.future_ee_goal_steps, device=self.device)
        self.future_ee_goal_cart = torch.zeros((self.num_envs, self.future_ee_goal_steps, 3), dtype=torch.float, device=self.device)
        self.future_ee_goal_sphere = torch.zeros((self.num_envs, self.future_ee_goal_steps, 3), dtype=torch.float, device=self.device)
        self.future_ee_goal_init = False

        

        assert(self.cfg.goal_ee.command_mode in ['cart', 'sphere'])

        if self.cfg.goal_ee.command_mode == 'cart':
            self.curr_ee_goal = self.curr_ee_goal_cart
        else:
            self.curr_ee_goal = self.curr_ee_goal_sphere

        local_axis_z_offset = self.cfg.goal_ee.local_axis_z_offset
        self.local_axis_z = torch.tensor(local_axis_z_offset, device=self.device).repeat(self.num_envs, 1)
        self.z_invariant_offset = torch.tensor(local_axis_z_offset,device=self.device).repeat(self.num_envs, 1)  
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_quat = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)


        self.base_align_z_axis = torch.tensor([0.,0.,local_axis_z_offset],dtype=torch.float,device=self.device).repeat(self.num_envs,1)
        self._gripper_state = self.rigid_body_states[:, self.gripperMover_handles][:, 0:13]
        self._gripper_pos = self.rigid_body_states[:, self.gripperMover_handles][:, 0:3]
        self._gripper_quat = self.rigid_body_states[:, self.gripperMover_handles][:, 3:7]
        self._local_gripper_pos = torch.zeros((self.num_envs,3),dtype=torch.float,device=self.device)  

    def refresh_ee_goal_variable(self):
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.base_align_z_axis = torch.cat([self.root_states[:, :2], self.local_axis_z], dim=1)

        self.base_quat = self.root_states[:, 3:7]
        base_yaw = get_euler_xyz(self.base_quat)[2]
        self.base_yaw_fixed = wrap_to_pi(base_yaw).view(self.num_envs,1)
        self.base_yaw_quat[:] = quat_from_euler_xyz(torch.tensor(0), torch.tensor(0), base_yaw)
        self.base_yaw_eular = torch.cat([torch.zeros(self.num_envs, 2, device=self.device), base_yaw.view(-1, 1)], dim=1)
        self._local_gripper_pos = quat_rotate_inverse(self.base_yaw_quat,self._gripper_pos - self.base_align_z_axis) 
        self.base_to_obj_dist = self._cube_object_pos[:,:2] - self.root_states[:,:2]
        self.base_align_z_axis[:,:2] = self.root_states[:, :2]

    def cart2sphere(self,cart):
        sphere = torch.zeros_like(cart)
        sphere[:, 0] = torch.norm(cart, dim=-1)
        sphere[:, 1] = torch.atan2(cart[:, 2], cart[:, 0])
        sphere[:, 2] = torch.asin(cart[:, 1] / sphere[:, 0])
        return sphere

    def sphere2cart(self,sphere):
        cart = torch.zeros_like(sphere)
        cart[:, 0] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.cos(sphere[:, 1])
        cart[:, 1] = sphere[:, 0] * torch.sin(sphere[:, 2])
        cart[:, 2] = sphere[:, 0] * torch.cos(sphere[:, 2]) * torch.sin(sphere[:, 1])
        return cart
    
    def update_curr_ee_goal(self):
        # self.refresh_ee_goal_variable()
        # t = torch.clip(self.goal_timer / self.traj_timesteps, 0, 1)
        # self.curr_ee_goal_sphere[:] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        # self.curr_ee_goal_cart[:] = self.sphere2cart(self.curr_ee_goal_sphere)
        # self.goal_timer += 1
        # resample_id = (self.goal_timer > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        # self._resample_ee_goal(resample_id)

        # #update current ee goal global axis
        # self._cube_object_pos = self.base_align_z_axis + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        # with future ee goal
        self.refresh_ee_goal_variable()
        t = torch.clip((self.goal_timer + self.future_ee_goal_dt * (self.future_ee_goal_steps-1))/ self.traj_timesteps, 0, 1)

        self.future_ee_goal_sphere = torch.roll(self.future_ee_goal_sphere, -1, dims=1)
        self.future_ee_goal_cart = torch.roll(self.future_ee_goal_cart, -1, dims=1)
        self.future_ee_goal_sphere[:, -1] = torch.lerp(self.ee_start_sphere, self.ee_goal_sphere, t[:, None])
        self.future_ee_goal_cart[:, -1] = self.sphere2cart(self.future_ee_goal_sphere[:, -1])
        resample_id = (self.goal_timer[:] + self.future_ee_goal_dt * (self.future_ee_goal_steps-1) > self.traj_total_timesteps).nonzero(as_tuple=False).flatten()
        self._resample_ee_goal(resample_id)
            
        self.curr_ee_goal_sphere[:] = self.future_ee_goal_sphere[:, 0]
        self.curr_ee_goal_cart[:] = self.future_ee_goal_cart[:, 0]
        self.goal_timer += 1

        #update current ee goal global axis
        self._cube_object_pos = self.base_align_z_axis + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

    def collision_check(self, env_ids):
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[env_ids, ..., None], self.ee_goal_sphere[env_ids, ...,  None], self.collision_check_t).squeeze(-1)
        ee_target_cart = self.sphere2cart(torch.permute(ee_target_all_sphere, (2, 0, 1)).reshape(-1, 3)).reshape(self.num_collision_check_samples, -1, 3)
        collision_mask = torch.any(torch.logical_and(torch.all(ee_target_cart < self.collision_upper_limits, dim=-1), torch.all(ee_target_cart > self.collision_lower_limits, dim=-1)), dim=0)
        underground_mask = torch.any(ee_target_cart[..., 2] < self.underground_limit, dim=0)
        return collision_mask | underground_mask
    
    def _get_init_start_ee_sphere(self):
        init_start_ee_cart = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        init_start_ee_cart[:, 0] = 0.15
        init_start_ee_cart[:, 2] = 0.15
        self.init_start_ee_sphere = self.cart2sphere(init_start_ee_cart)

    def _resample_ee_goal_sphere_once(self, env_ids):
        self.ee_goal_sphere[env_ids, 0] = torch_rand_float(self.goal_ee_l_ranges[0], self.goal_ee_l_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 1] = torch_rand_float(self.goal_ee_p_ranges[0], self.goal_ee_p_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.ee_goal_sphere[env_ids, 2] = torch_rand_float(self.goal_ee_y_ranges[0], self.goal_ee_y_ranges[1], (len(env_ids), 1), device=self.device).squeeze(1)
    
    def _resample_ee_goal_orn_once(self, env_ids):
        ee_goal_delta_orn_r = torch_rand_float(self.goal_ee_delta_orn_ranges[0, 0], self.goal_ee_delta_orn_ranges[0, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_p = torch_rand_float(self.goal_ee_delta_orn_ranges[1, 0], self.goal_ee_delta_orn_ranges[1, 1], (len(env_ids), 1), device=self.device)
        ee_goal_delta_orn_y = torch_rand_float(self.goal_ee_delta_orn_ranges[2, 0], self.goal_ee_delta_orn_ranges[2, 1], (len(env_ids), 1), device=self.device)
        self.ee_goal_delta_orn_euler[env_ids] = torch.cat([ee_goal_delta_orn_r, ee_goal_delta_orn_p, ee_goal_delta_orn_y], dim=-1)
        self.ee_goal_orn_euler[env_ids] = wrap_to_pi(self.ee_goal_delta_orn_euler[env_ids] + self.base_yaw_eular[env_ids])

    def _resample_ee_goal(self, env_ids, is_init=False):
        if len(env_ids) > 0:
            init_env_ids = env_ids.clone()
            self._resample_ee_goal_orn_once(env_ids)
            # if is_init:
            #     self.ee_start_sphere[env_ids] = self.init_start_ee_sphere[env_ids].clone()
            #     self._resample_ee_goal_sphere_once(env_ids, self.cfg.goal_ee.init_ranges)
            # else:
            self.ee_start_sphere[env_ids] = self.ee_goal_sphere[env_ids].clone()
            for i in range(10):
                self._resample_ee_goal_sphere_once(env_ids)
                collision_mask = self.collision_check(env_ids)
                env_ids = env_ids[collision_mask]
                if len(env_ids) == 0:
                    break
            self.ee_goal_cart[init_env_ids, :] = self.sphere2cart(self.ee_goal_sphere[init_env_ids, :])
            self.goal_timer[init_env_ids] = 0.0
            self.traj_timesteps[init_env_ids] = torch_rand_float(self.cfg.goal_ee.traj_time[0], self.cfg.goal_ee.traj_time[1], (len(init_env_ids), 1), device=self.device).squeeze() / self.dt

    def _draw_ee_goal_track(self):
        sphere_geom = gymutil.WireframeSphereGeometry(0.005, 8, 8, None, color=(1, 0, 0))

        t = torch.linspace(0, 1, 10, device=self.device)[None, None, None, :]
        ee_target_all_sphere = torch.lerp(self.ee_start_sphere[..., None], self.ee_goal_sphere[..., None], t).squeeze()
        ee_target_all_cart_world = torch.zeros_like(ee_target_all_sphere)
        for i in range(10):
            ee_target_cart = self.sphere2cart(ee_target_all_sphere[..., i])
            ee_target_all_cart_world[..., i] = quat_apply(self.base_yaw_quat, ee_target_cart)
        ee_target_all_cart_world += torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)[:, :, None]
        # curr_ee_goal_cart_world = quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart) + self.root_states[:, :3]
        for i in range(self.num_envs):
            for j in range(10):
                pose = gymapi.Transform(gymapi.Vec3(ee_target_all_cart_world[i, 0, j], ee_target_all_cart_world[i, 1, j], ee_target_all_cart_world[i, 2, j]), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], pose)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(1, 1, 0))
        transformed_target_ee = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self.curr_ee_goal_cart)

        sphere_geom_3 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 1, 1))
        upper_arm_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1)

        sphere_geom_2 = gymutil.WireframeSphereGeometry(0.05, 4, 4, None, color=(0, 0, 1))

        # cone = WireframeArrowGeometry()
        # gymutil.draw_lines(cone, self.gym, self.viewer, self.envs[0], gymapi.Transform(gymapi.Vec3(5, 5, 5), r=None))
        

        ee_pose = torch.cat([self.root_states[:, :2], self.z_invariant_offset], dim=1) + quat_apply(self.base_yaw_quat, self._local_gripper_pos)
        sphere_geom_origin = gymutil.WireframeSphereGeometry(0.1, 8, 8, None, color=(0, 1, 0))
        sphere_pose = gymapi.Transform(gymapi.Vec3(0, 0, 0), r=None)

        for i in range(self.num_envs):
            sphere_pose = gymapi.Transform(gymapi.Vec3(transformed_target_ee[i, 0], transformed_target_ee[i, 1], transformed_target_ee[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 
            
            sphere_pose_2 = gymapi.Transform(gymapi.Vec3(ee_pose[i, 0], ee_pose[i, 1], ee_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_2, self.gym, self.viewer, self.envs[i], sphere_pose_2) 

            sphere_pose_3 = gymapi.Transform(gymapi.Vec3(upper_arm_pose[i, 0], upper_arm_pose[i, 1], upper_arm_pose[i, 2]), r=None)
            gymutil.draw_lines(sphere_geom_3, self.gym, self.viewer, self.envs[i], sphere_pose_3) 










    #------------ reward functions----------------
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
        dis_err = torch.sum(torch.square(self._local_gripper_pos-self.curr_ee_goal), dim=1)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return torch.exp(-dis_err/0.1)
    
    def _reward_object_distance_l2(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Reward for lifting the object off the table."""
        dis_err = torch.sum(torch.square(self._local_gripper_pos-self.curr_ee_goal), dim=1)
        #dis_err = torch.sum(torch.square(self._gripper_pos-self.robot_root_states[:,0:3]+torch.tensor([0.5, 0.3, 0.4],device=self.device)), dim=1)
        #print("_object_distance:",dis_err,"value:",torch.exp(-dis_err/self.cfg.rewards.object_sigma).shape)  #[0.7~3.5]
        return dis_err