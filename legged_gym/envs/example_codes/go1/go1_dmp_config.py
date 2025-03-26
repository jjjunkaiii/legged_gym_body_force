from legged_gym.envs.go1.go1_config import Go1CfgPPO, Go1Cfg


class Go1DmpCfg(Go1Cfg):
    class env:
        num_envs = 4096 * 2
        num_actions = 12
        num_extra_observations = 12
        num_observations = Go1Cfg.env.num_observations + num_actions + num_extra_observations + 160
        num_privileged_obs = None
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20

    class cpg_policy:
        policy_path = 'logs/bio_cpg_go1_omi'
        cpg_obs_num = Go1Cfg.env.num_observations
        cpg_action_num = Go1Cfg.env.num_actions
        decimation = Go1Cfg.control.decimation

        class normalization(Go1Cfg.normalization):
            action_scale = 0.25

        class noise(Go1Cfg.noise):
            pass

    class normalization(Go1Cfg.normalization):
        clip_actions = 25.0


    class terrain(Go1Cfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        measure_heights = True
        measured_points_x = [0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525, 0.475, 0.425, 0.375,
                             0.325, 0.275, 0.225, 0.175, 0.125]  # 0.5mx0.8m rectangle (without center line)
        measured_points_y = [-0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225]
        measured_points_x_1 = [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                               0.8]  # 1mx1.6m rectangle (without center line)
        measured_points_y_1 = [-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
        # terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        num_rows = 20
        terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.4, 0.0]

    class control(Go1Cfg.control):
        action_scale = 0.01

    class rewards:
        class scales:
            tracking_lin_vel = 5.0
            tracking_ang_vel = 5.0
            collision = -1.

            # feet_air_time = 0.01 / 200000
            # fly = -5
            energy = -0.002

            # feet_stumble = -1.0
            action_xy = -0.01
            base_height = -50
            # feet_contact_forces = -0.01

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 180.  # changed from 180 to 150 forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32  # 25

    class commands(Go1Cfg.commands):
        heading_command = False

        class ranges(Go1Cfg.commands.ranges):
            lin_vel_x = [0.5, 1.2]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-0.5, 0.5]

    class noise(Go1Cfg.noise):
        class noise_scales(Go1Cfg.noise.noise_scales):
            height_measurements = 0.05

    class domain_rand(Go1Cfg.domain_rand):
        rand_map_update = False
        map_update_range = [0, 15]
        randomize_lag_timesteps = True
        lag_timesteps = 10

    class task_for_student:
        use_student = True
        add_noise = True
        add_delay = True
        delay_count = 66

        class noise:
            noise_level = 1.0

            class noise_scales:
                dof_pos = 0.03
                dof_vel = 1.5
                lin_vel = 0.2
                ang_vel = 1.2
                gravity = 0.1
                height_measurements = 0.05

        rand_map_update = True
        map_update_range = [1, 15]


class Go1DmpCfgPPO(Go1CfgPPO):
    class runner(Go1CfgPPO.runner):
        policy_class_name = 'ActorCriticEncoder'
        experiment_name = 'dmp_go1_rma_encoder'
        run_name = 'encoder_gru_delay_action_low_reward'
        max_iterations = 20000  # number of policy updates
        save_interval = 200

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    class policy(Go1CfgPPO.policy):
        encoder_input_dim = 12
        encoder_output_dim = 22
        encoder_map_input_dim = 160
        encoder_map_output_dim = 50
        rnn_type = 'gru'
        use_student = True
        student_rnn_type = 'gru'
        student_map_rnn_type = 'lstm'
