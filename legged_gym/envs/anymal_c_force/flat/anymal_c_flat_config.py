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

from legged_gym.envs import AnymalCRoughForceCfg, AnymalCRoughForceCfgPPO

class AnymalCFlatForceCfg( AnymalCRoughForceCfg ):
    class env( AnymalCRoughForceCfg.env ):
        num_observations = 51
        num_envs = 4096
  
    class terrain( AnymalCRoughForceCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( AnymalCRoughForceCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( AnymalCRoughForceCfg.rewards ):
        max_contact_force = 350.
        base_height_target = 0.5
        only_positive_rewards = True
        force_sigma = 20.
        tracking_sigma = 0.25
        class scales:#( AnymalCRoughForceCfg.rewards.scales )
            orientation = -10.0
            torques = -0.000025
            # lin_vel_z = -2.0
            ang_vel_xy = -0.1
            dof_acc = -2.5e-7
            base_height = -0. 
            collision = -1.
            action_rate = -0.1
            force_tracking = 5.0
            stay_still_lin = 1.0
            stay_still_ang = 0.5
    
    class commands( AnymalCRoughForceCfg.commands ):
        num_commands = 3
        class ranges:
            # f_x = [-50, 50] # min max [N]
            # f_y = [-50, 50]   # min max [N]
            # f_z = [-1, 1]
            f_x = [-0.01, 0.01]
            f_y = [-0.01, 0.01]
            f_z = [-0.01, 0.01]

    class domain_rand( AnymalCRoughForceCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.

    class normalization( AnymalCRoughForceCfg.normalization):
        class obs_scales( AnymalCRoughForceCfg.normalization.obs_scales):
            commands = 0.01
            reactional_forces = 0.01


class AnymalCFlatForceCfgPPO( AnymalCRoughForceCfgPPO ):
    class policy( AnymalCRoughForceCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( AnymalCRoughForceCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( AnymalCRoughForceCfgPPO.runner):
        run_name = 'track-zero-force-command'
        experiment_name = 'flat_anymal_c_force'
        load_run = -1
        max_iterations = 1000#300
