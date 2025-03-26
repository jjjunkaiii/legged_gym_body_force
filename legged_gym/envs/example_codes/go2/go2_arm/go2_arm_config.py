from legged_gym.envs.go2.go2_torque.go2_torque_config import GO2TorqueCfg, GO2TorqueCfgPPO

class Go2ArmCfg(GO2TorqueCfg):
    class env(GO2TorqueCfg.env):
        num_observations = 62 - 5 + 13 + 16 + 2 # -commands +last action +arm pos/vel
        num_actions = 13 # vx, vy, vyaw, pith, height, joint action
        target_pos = [0.0, 0.0, 0.0]  # 目标点，默认在原点

    class init_state(GO2TorqueCfg.init_state):
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,  # [rad]
            'RL_hip_joint': 0.1,  # [rad]
            'FR_hip_joint': -0.1,  # [rad]
            'RR_hip_joint': -0.1,  # [rad]

            'FL_thigh_joint': 1.45,  # [rad]
            'RL_thigh_joint': 1.45,  # [rad]
            'FR_thigh_joint': 1.45,  # [rad]
            'RR_thigh_joint': 1.45,  # [rad]

            'FL_calf_joint': -2.5,  # [rad]
            'RL_calf_joint': -2.5,  # [rad]
            'FR_calf_joint': -2.5,  # [rad]
            'RR_calf_joint': -2.5,  # [rad]
        }
        for i in range(1, 9):
            default_joint_angles['joint' + str(i)] = 0.0

    class asset(GO2TorqueCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2/urdf/go2_X5_r.urdf'

    class control(GO2TorqueCfg.control):
        control_type = 'T'

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 1.5
            target_pos = 0.1

    class commands(GO2TorqueCfg.commands):
        num_commands = 2
        target_command = True

        class ranges:
            target_point_x = [-5,5]  # min max [m]
            target_point_y = [-5,5]  # min max [m]

    class low_level_policy:
        policy_path = '/home/marmot/CHANG Linnan/Legged_pitch/logs/go2_new_start/exported/policies/policy_1.pt'
        cfg = GO2TorqueCfg

    class rewards(GO2TorqueCfg.rewards):
        target_tracking_sigma = 0.5

        class scales:
            reach_target = 5
            pass

class Go2ArmCfgPPO(GO2TorqueCfgPPO):
    class runner(GO2TorqueCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        run_name = 'with_X5'
        experiment_name = 'go2_arm'
        max_iterations = 3000