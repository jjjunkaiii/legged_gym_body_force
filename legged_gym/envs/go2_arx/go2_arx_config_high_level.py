# from legged_gym.envs.go2_arx.go2_arx_config import Go2ArxRoughCfg, Go2ArxRoughCfgPPO
from .go2_arx_config import Go2ArxRoughCfg, Go2ArxRoughCfgPPO
import numpy as np
class Go2ArxHLRoughCfg( Go2ArxRoughCfg ):

    class env( Go2ArxRoughCfg.env ):
        num_envs = 4096
        num_observations = 79#TODO
        symmetric = False  #true :  set num_privileged_obs = None;    false: num_privileged_obs = observations + 187 ,set "terrain.measure_heights" to true
        num_privileged_obs = num_observations + 187 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 7 #linvel x, y, yaw, heading, l, y, p
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class policy:
        policy_path = 'logs/go2_arx/Feb21_16-09-01_First_Run'
        policy_name = 'policy_1.pt'

    class rewards( Go2ArxRoughCfg.rewards ):
        class scales( Go2ArxRoughCfg.rewards.scales ):
            termination = -1
            # tracking_lin_vel = 2.0
            # tracking_ang_vel = 0.5
            # lin_vel_z = -0.0
            # ang_vel_xy = -0.1
            # orientation = -0.5
            torques = -0.0001
            dof_vel = -0.
            dof_acc = -2.5e-9
            base_height = -0.2
            feet_air_time =  1.0
            collision = -1.
            feet_stumble = -0.0 
            action_rate = -0.001
            stand_still = -0.
            dof_pos_limits =-10.
            object_distance = 2.
            object_distance_l2=-1
            # torque_limits = -0.0005

        only_positive_rewards = True # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9 # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.25
        max_contact_force = 100. # forces above this value are penalized
    
    class commands(Go2ArxRoughCfg.commands ):
        curriculum = False
        max_curriculum = 1.
        num_commands = 3 # x, y, z
        resampling_time = 0.1 # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        class ranges:
            x = [-2, 2]#[-1.0, 1.0] # min max [m/s]
            y = [-2., 2]   # min max [m/s]
            z = [0.2, 1.5]    # min max [rad/s]

    class normalization(Go2ArxRoughCfg.normalization):
        class obs_scales(Go2ArxRoughCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0
            gripper_track = 1.0
            commands = 1.0
        clip_observations = 100.
        # clip_actions = 100.
        action_upper = [100., 100., 150., 100., 100., 100., 100.]
        action_lower = [-100., -100., -20., -100., -100., -100., -100.]

    class control(Go2ArxRoughCfg.control):
        action_scale = 0.02


class Go2ArxHLRoughCfgPPO( Go2ArxRoughCfgPPO ):
    class algorithm( Go2ArxRoughCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( Go2ArxRoughCfgPPO.runner ):
        run_name = 'test_run'
        experiment_name = 'go2_arx_HL'

  
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24 # per iteration
        max_iterations = 1000 # number of policy updates

        # logging
        save_interval = 50 # check for potential saves every this many iterations

        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 #= last saved model
        resume_path = None # updated from load_run and chkpt