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

from legged_gym.envs import AnymalCRoughHybridCfg, AnymalCRoughHybridCfgPPO

class AnymalCFlatHybridCfg( AnymalCRoughHybridCfg ):
    class env( AnymalCRoughHybridCfg.env ):
        num_observations = 69#TODO
        num_envs = 4096
        episode_length_s = 10.0
  
    class terrain( AnymalCRoughHybridCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( AnymalCRoughHybridCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( AnymalCRoughHybridCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.5
        only_positive_rewards = True
        force_sigma = 20.
        tracking_sigma = 0.25
        class scales:#( AnymalCRoughHybridCfg.rewards.scales )
            orientation = -10.0
            torques = -0.000025
            # lin_vel_z = -2.0
            ang_vel_xy = -0.1
            dof_acc = -2.5e-7
            base_height = -0. 
            collision = -1.
            action_rate = -0.1
            # force_tracking = 10.0
            position_tracking = 5.0
            stay_still_lin_z = 1.0
            stay_still_ang = 0.5
    
    class commands( AnymalCRoughHybridCfg.commands ):
        num_commands = 4
        force_curriculum = True
        resampling_time = 0.05
        class ranges:
            f = [-50, 50] # min max [N]

    class domain_rand( AnymalCRoughHybridCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

    class normalization( AnymalCRoughHybridCfg.normalization):
        class obs_scales( AnymalCRoughHybridCfg.normalization.obs_scales):
            commands_p = 1.0
            force = 0.01
            setpoint_vel = 1.0


class AnymalCFlatHybridCfgPPO( AnymalCRoughHybridCfgPPO ):
    class policy( AnymalCRoughHybridCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( AnymalCRoughHybridCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( AnymalCRoughHybridCfgPPO.runner):
        run_name = 'rm-force;follow-straght-line;mod-rew:track-full-pos-diff'
        experiment_name = 'flat_anymal_c_hybrid'
        load_run = -1
        max_iterations = 10000#1000#300
