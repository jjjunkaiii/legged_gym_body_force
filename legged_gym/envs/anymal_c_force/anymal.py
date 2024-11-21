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
from .mixed_terrains.anymal_c_rough_config import AnymalCRoughForceCfg

# from legged_gym.utils.draw_arrow import WireframeArrowGeometry

class Anymalforce(LeggedRobot):
    cfg : AnymalCRoughForceCfg
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
        self.init_base_pos = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
        init_pos = torch.tensor(self.cfg.init_state.pos, dtype=torch.float32, device=self.device)
        for i in range(self.num_envs):
            self.init_base_pos[i] = init_pos + self.env_origins[i]

        self.setpoint_pos = torch.clone(self.init_base_pos)
        self.setpoint_vel = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)

        self.reactional_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
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
        command_mode = 'random' # 'random' or 'circular'

        if self.cfg.commands.force_curriculum:
            coef = self._get_training_progress()
        else:
            coef = 1.
        # self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["f_x"][0], self.command_ranges["f_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)*coef
        # self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["f_y"][0], self.command_ranges["f_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)*coef
        # self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["f_z"][0], self.command_ranges["f_z"][1], (len(env_ids), 1), device=self.device).squeeze(1)*coef
        sampled_command_x = torch_rand_float(self.command_ranges["f_x"][0], self.command_ranges["f_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        sampled_command_y = torch_rand_float(self.command_ranges["f_y"][0], self.command_ranges["f_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        sampled_command_z = torch_rand_float(self.command_ranges["f_z"][0], self.command_ranges["f_z"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        max_sampled_command_x = self.command_ranges["f_x"][1] * torch.ones(len(env_ids), device=self.device)
        max_sampled_command_y = self.command_ranges["f_y"][1] * torch.ones(len(env_ids), device=self.device)

        if command_mode == 'random':
            self.commands[env_ids, 0] = sampled_command_x*coef
            self.commands[env_ids, 1] = sampled_command_y*coef
            self.commands[env_ids, 2] = sampled_command_z*coef
        elif command_mode == 'circular':
            print("circular")
            omega = 0.2 * torch.ones(len(env_ids), device=self.device)
            self.commands[env_ids, 0] = max_sampled_command_x * torch.cos(omega * self.common_step_counter * self.dt)
            self.commands[env_ids, 1] = max_sampled_command_y * torch.sin(omega * self.common_step_counter * self.dt)
            self.commands[env_ids, 2] = 0

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
        #-------------------
        self.pos_diff = self.root_states[:, :3] - self.setpoint_pos
        self.vel_diff = self.root_states[:, 7:10] - self.setpoint_vel
        #-------------------

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
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.reactional_forces), gymtorch.unwrap_tensor(self.reactional_torques), gymapi.ENV_SPACE)

    def _compute_force(self):
        self.reactional_forces[:,0,:]=-(self.pos_diff[:] * 700 + self.vel_diff[:] * 6)

    def _reset_force(self, env_ids):
        self.reactional_forces[env_ids,0] = torch.zeros(3, device=self.device, dtype=torch.float, requires_grad=False)

    # 4. visualize force vectors
    def _draw_debug_vis_force_vector(self):
        self.gym.clear_lines(self.viewer)
        # draw force vectors, for each environment, draw commanded force vector in yellow and reactional force vector in green, from init_base_pos in each environment
        sphere_geom_cmd = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(1, 1, 0))
        sphere_geom_react = gymutil.WireframeSphereGeometry(0.1, 4, 4, None, color=(0, 1, 0))
        for i in range(self.num_envs):
            start = self.setpoint_pos[i]
            end_cmd = start - self.commands[i] * 0.05
            end_react = start - self.reactional_forces[i,0] * 0.05
            # print(end_react)
            gymutil.draw_line(gymapi.Vec3(*start.cpu().numpy()), gymapi.Vec3(*end_cmd.cpu().numpy()), gymapi.Vec3(1, 1, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_line(gymapi.Vec3(*start.cpu().numpy()), gymapi.Vec3(*end_react.cpu().numpy()), gymapi.Vec3(0, 1, 0), self.gym, self.viewer, self.envs[i])
            gymutil.draw_lines(sphere_geom_cmd, self.gym, self.viewer, self.envs[i], gymapi.Transform(gymapi.Vec3(*end_cmd.cpu().numpy())))
            gymutil.draw_lines(sphere_geom_react, self.gym, self.viewer, self.envs[i], gymapi.Transform(gymapi.Vec3(*end_react.cpu().numpy())))
            #time.sleep(0.05)
            gymapi.Vec3(*end_react.cpu().numpy())
     
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
                                    quat_rotate_inverse(self.base_quat, self.commands)*self.obs_scales.commands, 
                                    quat_rotate_inverse(self.base_quat, self.reactional_forces[:,0,:])*self.obs_scales.reactional_forces,
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
        force_error_x = torch.abs(self.commands[:,0] - self.reactional_forces[:,0,0])
        force_error_y = torch.abs(self.commands[:,1] - self.reactional_forces[:,0,1])
        force_error_z = torch.abs(self.commands[:,2] - self.reactional_forces[:,0,2])
        rew_force = torch.exp(-torch.sqrt(force_error_x**2+force_error_y**2+force_error_z**2)/ self.cfg.rewards.force_sigma)

        return rew_force
    
    def _reward_stay_still_lin(self):
        """ Reward for staying still
        """
        lin_vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
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
        progress = torch.tanh(torch.tensor(self.common_step_counter / (1000 * 24 * 2), device=self.device))
        progress = progress ** 2
        return progress