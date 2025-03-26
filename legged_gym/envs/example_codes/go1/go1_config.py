from legged_gym.envs.base.base_config import BaseConfig


class Go1Cfg(BaseConfig):
    class env:
        num_envs = 4096
        num_observations = 76
        num_privileged_obs = None
        num_actions = 12
        env_spacing = 3.
        send_timeouts = True
        episode_length_s = 20

    class terrain:
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        show_terrain = False
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.
        # rough terrain only:
        measure_heights = False
        measured_points_x = [0.875, 0.825, 0.775, 0.725, 0.675, 0.625, 0.575, 0.525, 0.475, 0.425, 0.375,
                             0.325, 0.275, 0.225, 0.175, 0.125]  # 0.5mx0.8m rectangle (without center line)
        measured_points_y = [-0.225, -0.175, -0.125, -0.075, -0.025, 0.025, 0.075, 0.125, 0.175, 0.225]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5  # starting curriculum state
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class rewards:
        class scales:
            tracking_lin_vel_x = 0.75
            tracking_lin_vel_y = 0.75
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            energy = -0.001

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        max_contact_force = 180.  # forces above this value are penalized
        soft_dof_pos_limit = 0.9
        base_height_target = 0.32  # 25

    class control:
        control_type = 'CPG'
        use_actuator_net = False
        stiffness = {'joint': 100.}  # [N*m/rad]
        damping = {'joint': 2.0}  # [N*m*s/rad]
        action_scale = 0.25
        decimation = 20
        freq_max = 40
        freq_low = -5

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            height_measurements = 5.0

        clip_observations = 100.
        clip_actions = 100  # 4 #100.

    class commands:
        curriculum = False
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5  # time before command are changed[s]
        heading_command = False  # True # if true: compute ang vel command from heading error
        max_vel_x = 2.0
        max_curriculum = 2.0

        class ranges:
            lin_vel_x = [0.5, 2]  # min max [m/s]
            lin_vel_y = [-0.2, 0.2]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-0.5, 0.5]

    class domain_rand:
        rand_interval_s = 10
        push_interval_s = 15
        gravity_rand_interval_s = 7
        gravity_impulse_duration = 1.0

        randomize_rigids_after_start = True
        randomize_base_mass = True
        added_mass_range = [-1., 6.]  # [kg]
        randomize_com_displacement = True
        com_displacement_range_x = [-0.08, 0.08]  # [m]
        com_displacement_range_y = [-0.05, 0.05]  # [m]
        com_displacement_range_z = [-0.02, 0.02]  # [m]
        randomize_friction = True
        friction_range = [0.2, 1.5]
        randomize_restitution = False
        restitution_range = [0, 1.0]
        randomize_gravity = False
        gravity_range = [-1.0, 1.0]  # [m/s^2]
        push_robots = True
        max_push_vel_xy = 0.03  # [m/s]

        randomize_reset_dof_pos = False
        randomize_motor_strength = False
        motor_strength_range = [0.9, 1.1]
        randomize_Kp_factor = True
        Kp_factor_range = [0.8, 1.3]
        randomize_Kd_factor = True
        Kd_factor_range = [0.5, 1.5]
        randomize_motor_offset = False
        motor_offset_range = [-0.1, 0.1]
        randomize_decimation = False
        decimation_range = [20, 20]

    class noise:
        add_noise = True  # True
        noise_level = 1.0  # scales other values

        class noise_scales:
            dof_pos = 0.03
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.4  # need bigger
            gravity = 0

            # dof_pos = 0.03
            # dof_vel = 1.5
            # lin_vel = 0.2
            # ang_vel = 1.2
            # gravity = 0.1
            # height_measurements = 0.05

    class sim:
        dt = 0.001  # 0.005
        substeps = 1
        gravity = [0., 0., -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2 ** 23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = 2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)

    class viewer:
        ref_env = 0
        pos = [8, -6, 6]  # [m]
        lookat = [8., 2, 1.]  # [m]

    class asset:
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False  # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up

        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1.urdf'
        foot_name = "foot"
        name = "go1"
        penalize_contacts_on = ["thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 1  #
        hip_link_length_a1 = 0.0838
        thigh_link_length_a1 = 0.213
        calf_link_length_a1 = 0.213

        hip_link_length_go1 = 0.08
        thigh_link_length_go1 = 0.213
        calf_link_length_go1 = 0.213

    class init_state:
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]

        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.,  # [rad]
            'RL_hip_joint': 0.,  # [rad]
            'FR_hip_joint': 0.,  # [rad]
            'RR_hip_joint': 0.,  # [rad]

            'FL_thigh_joint': 0.785,  # [rad]
            'RL_thigh_joint': 0.785,  # [rad]
            'FR_thigh_joint': 0.785,  # [rad]
            'RR_thigh_joint': 0.785,  # [rad]

            'FL_calf_joint': -1.57,  # [rad]
            'RL_calf_joint': -1.57,  # [rad]
            'FR_calf_joint': -1.57,  # [rad]
            'RR_calf_joint': -1.57,  # [rad]
        }


class Go1CfgPPO(BaseConfig):
    seed = 1
    runner_class_name = 'OnPolicyRunner'

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    class algorithm:
        # training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.e-3
        schedule = 'adaptive'  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24  # 64 # per iteration
        max_iterations = 3000  # number of policy updates
        # logging
        save_interval = 50  # check for potential saves every this many iterations
        experiment_name = 'bio_cpg_go1_omi'
        run_name = 'random_decimation'

        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt
