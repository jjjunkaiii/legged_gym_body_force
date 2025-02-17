# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# from torch.tensor import Tensor
from typing import Tuple, Dict

from legged_gym.envs import LeggedRobot
from legged_gym import LEGGED_GYM_ROOT_DIR
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughHybridCfg

# from legged_gym.utils.draw_arrow import WireframeArrowGeometry

class Anymalhybrid(LeggedRobot):
    cfg : AnymalCRoughHybridCfg
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self.debug_viz = True
        # load actuator network
        if self.cfg.control.use_actuator_network:
            actuator_network_path = self.cfg.control.actuator_net_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            self.actuator_network = torch.jit.load(actuator_network_path).to(self.device)
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        # Additionaly empty actuator network hidden states
        self.sea_hidden_state_per_env[:, env_ids] = 0.
        self.sea_cell_state_per_env[:, env_ids] = 0.

    def _init_buffers(self):
        super()._init_buffers()
        # Additionally initialize actuator network hidden state tensors
        self.sea_input = torch.zeros(self.num_envs*self.num_actions, 1, 2, device=self.device, requires_grad=False)
        self.sea_hidden_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_cell_state = torch.zeros(2, self.num_envs*self.num_actions, 8, device=self.device, requires_grad=False)
        self.sea_hidden_state_per_env = self.sea_hidden_state.view(2, self.num_envs, self.num_actions, 8)
        self.sea_cell_state_per_env = self.sea_cell_state.view(2, self.num_envs, self.num_actions, 8)

        self.pos_diff = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.vel_diff = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.pos_diff_nor = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.pos_diff_tan = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.vel_diff_nor = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.vel_diff_tan = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.init_base_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        init_pos = torch.tensor(self.cfg.init_state.pos, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            self.init_base_pos[i] = init_pos + self.env_origins[i]
        # derivitive of the position commands, used to compute the commanded velocity, tangential and normal direction
        self.commands_dot = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.setpoint_pos = torch.clone(self.init_base_pos)
        self.setpoint_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.setpoint_vel_magnitude = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        self.setpoint_tan = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        self.setpoint_nor = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.reactional_forces = torch.zeros((self.num_envs, self.num_bodies), device=self.device, dtype=torch.float)
        self.reactional_forces_3d = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.reactional_torques = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

    def _compute_torques(self, actions):
        # Choose between pd controller and actuator network
        if self.cfg.control.use_actuator_network:
            with torch.inference_mode():
                self.sea_input[:, 0, 0] = (actions * self.cfg.control.action_scale + self.default_dof_pos - self.dof_pos).flatten()
                self.sea_input[:, 0, 1] = self.dof_vel.flatten()
                torques, (self.sea_hidden_state[:], self.sea_cell_state[:]) = self.actuator_network(self.sea_input, (self.sea_hidden_state, self.sea_cell_state))
            return torques
        else:
            # pd controller
            return super()._compute_torques(actions)  

#----------------CUSTOMISED ENVIRONMENT----------------
    # 1. change commands
    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        # force curriculum learning
        if self.cfg.commands.force_curriculum:
            coef = self._get_training_progress()
        else:
            coef = 1.
        # in total 4 commands: p_x, p_y, p_z, f 
        omega = 0.2 * torch.ones(len(env_ids), device=self.device)
        r = 0.05 * torch.ones(len(env_ids), device=self.device)
        max_sampled_force = self.command_ranges["f"][1] * torch.ones(len(env_ids), device=self.device)
        # random commands
        counter = self.episode_length_buf[env_ids]
        self.commands[env_ids, 0] = omega * counter * self.dt# r * torch.cos(omega * self.common_step_counter * self.dt) # p_x
        self.commands[env_ids, 1] = torch.zeros_like(self.commands[env_ids, 1], device=self.device) #r * torch.sin(omega * self.common_step_counter * self.dt) # p_y
        self.commands[env_ids, 2] = torch.zeros(len(env_ids), device=self.device) # p_z
        # force f is a scalar, the direction is always along the normal of the curve
        self.commands[env_ids, 3] = 0 * coef#0 * torch.sin(4 * omega * self.common_step_counter * self.dt) * max_sampled_force * coef

        # compute the commanded position
        self.setpoint_pos[env_ids] = self.commands[env_ids, :3] + self.init_base_pos[env_ids]
        # compute the derivitive of the position commands
        self.commands_dot[env_ids, 0] = -r * omega * torch.sin(omega * self.common_step_counter * self.dt) # p_x_dot
        self.commands_dot[env_ids, 1] = r * omega * torch.cos(omega * self.common_step_counter * self.dt) # p_y_dot
        self.commands_dot[env_ids, 2] = torch.zeros(len(env_ids), device=self.device) # p_z_dot
        # compute the velocity at commanded poistion
        self.setpoint_vel[env_ids] = self.commands_dot[env_ids]
        # compute the magnitude of the commanded velocity
        self.setpoint_vel_magnitude[env_ids] = torch.norm(self.setpoint_vel[env_ids], dim=-1)
        # compute the unit tangential and normal direction of the commanded velocity
        self.setpoint_tan[env_ids] = self.setpoint_vel[env_ids] / self.setpoint_vel_magnitude[env_ids].unsqueeze(1)
        self.setpoint_nor[env_ids] = torch.stack((-self.setpoint_tan[env_ids][:,1], self.setpoint_tan[env_ids][:,0], torch.zeros(len(env_ids), device=self.device)), dim=-1)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
    # 2. get descrepancy between initial position and current position
    # def _init_buffers(self)
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

        self._compute_diff()

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis_force_vector()

    def _compute_diff(self):
        self.pos_diff = self.root_states[:, :3] - self.setpoint_pos
        self.pos_diff_nor = torch.sum(self.pos_diff * self.setpoint_nor, dim=-1)
        self.pos_diff_tan = torch.sum(self.pos_diff * self.setpoint_tan, dim=-1)

        self.vel_diff = self.root_states[:, 7:10] - self.setpoint_vel
        self.vel_diff_nor = torch.sum(self.vel_diff * self.setpoint_nor, dim=-1)
        self.vel_diff_tan = torch.sum(self.vel_diff * self.setpoint_tan, dim=-1)

    # 3. add force
    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        # start_time = time()
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            #-------------------
            self._apply_force()
            #-------------------
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
        # print("Time for step: ", time()-start_time)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras
    
    def reset_idx(self, env_ids):
        """ Reset states, observations, rewards and done flags of some environments

        Args:
            env_ids (List[int]): List of environment ids to reset
        """
        super().reset_idx(env_ids)
        # self._reset_force(env_ids)

    def _apply_force(self):
        self._compute_force()
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.reactional_forces_3d), gymtorch.unwrap_tensor(self.reactional_torques), gymapi.ENV_SPACE)

    def _compute_force(self):
        self.reactional_forces[:,0]=-(self.pos_diff_nor[:] * 700 + self.vel_diff_nor[:] * 6)
        self.reactional_forces_3d[:,0,:] = torch.zeros_like(self.reactional_forces_3d[:,0,:])
        #self.reactional_forces[:,0].unsqueeze(1) * self.setpoint_nor

    # 4. visualize force vectors 
    def _draw_debug_vis_force_vector(self): #TODO draw the trajectory
        self.gym.clear_lines(self.viewer)
        # draw force vectors, for each environment, draw commanded force vector in yellow and reactional force vector in green, from init_base_pos in each environment
        sphere_geom_cmd = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        sphere_geom_react = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            start_setpoint = self.setpoint_pos[i]
            start_body = self.root_states[i,:3]
            end_cmd = start_setpoint - self.commands[i, 3] * self.setpoint_nor[i] * 0.2
            end_react = start_body - self.reactional_forces[i,0] * self.setpoint_nor[i] * 0.2
            gymutil.draw_line(gymapi.Vec3(*start_setpoint.cpu().numpy()), gymapi.Vec3(*end_cmd.cpu().numpy()), gymapi.Vec3(1, 1, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_line(gymapi.Vec3(*start_body.cpu().numpy()), gymapi.Vec3(*end_react.cpu().numpy()), gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_lines(sphere_geom_cmd, self.gym, self.viewer, self.envs[i], gymapi.Transform(gymapi.Vec3(*end_cmd.cpu().numpy())))
            gymutil.draw_lines(sphere_geom_react, self.gym, self.viewer, self.envs[i], gymapi.Transform(gymapi.Vec3(*end_react.cpu().numpy())))
     
    # 5. change observation space
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((  self.base_lin_vel * self.obs_scales.lin_vel,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions,
                                    quat_rotate_inverse(self.base_quat, self.commands[:,:3])*self.obs_scales.commands_p, 
                                    quat_rotate_inverse(self.base_quat, self.setpoint_vel)*self.obs_scales.setpoint_vel,
                                    (self.commands[:,3] * self.obs_scales.force).reshape(-1,1),
                                    self.reactional_forces * self.obs_scales.force,
                                    ),dim=-1)
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # 6. change reward function
    def _reward_force_tracking(self):
        """ Reward for tracking commanded forces
        """
        # force_error = torch.norm(self.commands - self.reactional_forces[:,0,:], dim=-1)
        # return torch.exp(-force_error / self.cfg.rewards.scales.force_tracking)
        force_error = torch.abs(self.commands[:,3] - self.reactional_forces[:,0])
        rew_force = torch.exp(-torch.sqrt(force_error**2)/ self.cfg.rewards.force_sigma)

        return rew_force
    
    def _reward_position_tracking(self):
        """ Reward for tracking commanded position at tangential direction
        """
        #pos_error = torch.abs(self.pos_diff_tan)
        pos_error = torch.abs(self.pos_diff).sum(dim=1)
        return torch.exp(-pos_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_stay_still_lin_z(self):
        """ Reward for staying still in z direction
        """
        z_pose_error = torch.abs(self.root_states[:, 2] - self.init_base_pos[:, 2])
        return torch.exp(-z_pose_error/self.cfg.rewards.tracking_sigma)

    
    def _reward_stay_still_ang(self):
        """ Reward for staying still
        """
        ang_vel_error = torch.square(self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
        
    # 7. change termination conditions (optional)

    # 8. other functions
    def _get_training_progress(self):
        """ Get training progress
        """
        # max_iter = self.training_cfg.runner.max_iterations
        # cur_iter = (self.common_step_counter - 1) // self.training_cfg.runner.num_steps_per_env
        # print(max_iter)
        # progress = cur_iter / max_iter
        # #print(f"Progress: {progress}")
        # return progress
        progress = torch.tanh(torch.tensor(self.common_step_counter / (1000 * 24 * 4), device=self.device))
        progress = progress ** 2
        return progress