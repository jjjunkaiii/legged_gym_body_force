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

from legged_gym.envs import AnymalCRoughCfg, AnymalCRoughCfgPPO

class AnymalCFlatCfg( AnymalCRoughCfg ):
    class env( AnymalCRoughCfg.env ):
        num_observations = 48
  
    class terrain( AnymalCRoughCfg.terrain ):
        mesh_type = 'plane'
        measure_heights = False
  
    class asset( AnymalCRoughCfg.asset ):
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards( AnymalCRoughCfg.rewards ):
        max_contact_force = 350.
        class scales ( AnymalCRoughCfg.rewards.scales ):
            # orientation = -5.0
            # torques = -0.000025
            # feet_contact_forces = -0.01
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            # lin_vel_z = -2.0
            ang_vel_xy = -0.5
            # dof_acc = -2.5e-7
            # collision = -1.
            # action_rate = -0.01
            termination = -1.0
            pass
    
    class commands( AnymalCRoughCfg.commands ):
        heading_command = False
        resampling_time = 4.
        # class ranges( AnymalCRoughCfg.commands.ranges ):
        #     ang_vel_yaw = [-1.5, 1.5]

    class domain_rand( AnymalCRoughCfg.domain_rand ):
        friction_range = [0., 1.5] # on ground planes the friction combination mode is averaging, i.e total friction = (foot_friction + 1.)/2.
        push_robots = True
        push_interval_s = 2#15
        max_push_vel_xy = 2.#1.
        
class AnymalCFlatCfgPPO( AnymalCRoughCfgPPO ):
    class policy( AnymalCRoughCfgPPO.policy ):
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [128, 64, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm( AnymalCRoughCfgPPO.algorithm):
        entropy_coef = 0.01

    class runner ( AnymalCRoughCfgPPO.runner):
        run_name = 'push, add rew: tracking_lin_vel = 0.2, iter=1000'
        experiment_name = 'flat_anymal_c'
        load_run = -1
        max_iterations = 1000
