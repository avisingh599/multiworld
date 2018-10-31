import gym
from gym.envs.registration import register
import logging

LOGGER = logging.getLogger(__name__)

_REGISTERED = False


def register_custom_envs():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    LOGGER.info("Registering multiworld mujoco gym environments")

    """
    Reaching tasks
    """

    register(
        id='SawyerReachXYEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz.sawyer_reach:SawyerReachXYEnv',
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'murtaza'
        },
        kwargs={
            'hide_goal_markers': True,
            'norm_order':2,
        },
    )

    register(
        id='Image48SawyerReachXYEnv-v0',
        entry_point=create_image_48_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'vitchyr'
        },
    )
    register(
        id='Image84SawyerReachXYEnv-v0',
        entry_point=create_image_84_sawyer_reach_xy_env_v0,
        tags={
            'git-commit-hash': '2d95c75',
            'author': 'vitchyr'
        },
    )

    """
    Pushing tasks, XY, With Reset
    """


    """
    Door Hook Env
    """

    register(
        id='SawyerDoorHookEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=False,
        )
    )

    register(
        id='SawyerDoorHookResetFreeEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_door_hook:SawyerDoorHookEnv',
        tags={
            'git-commit-hash': '15b48d5',
            'author': 'murtaza',
        },
        kwargs=dict(
            goal_low=(-0.1, 0.45, 0.1, 0),
            goal_high=(0.05, 0.65, .25, .83),
            hand_low=(-0.1, 0.45, 0.1),
            hand_high=(0.05, 0.65, .25),
            max_angle=.83,
            xml_path='sawyer_xyz/sawyer_door_pull_hook.xml',
            reward_type='angle_diff_and_hand_distance',
            reset_free=True,
        )
    )

    """
    Pick and Place
    """
    register(
        id='SawyerPickupEnv-v0',
        entry_point='multiworld.envs.mujoco.sawyer_xyz'
                    '.sawyer_pick_and_place:SawyerPickAndPlaceEnv',
        tags={
            'git-commit-hash': '30f23f7',
            'author': 'steven',
        },
        kwargs=dict(
            hand_low=(-0.1, 0.55, 0.05),
            hand_high=(0.0, 0.65, 0.2),
            action_scale=0.02,
            hide_goal_markers=True,
        )

    )


def create_image_48_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_image_84_sawyer_reach_xy_env_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_xyz_reacher_camera_v0

    wrapped_env = gym.make('SawyerReachXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        84,
        init_camera=sawyer_xyz_reacher_camera_v0,
        transpose=True,
        normalize=True,
    )


def create_Image48SawyerPushAndReacherXYEnv_v0():
    from multiworld.core.image_env import ImageEnv
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_top_down

    wrapped_env = gym.make('SawyerPushAndReacherXYEnv-v0')
    return ImageEnv(
        wrapped_env,
        48,
        init_camera=sawyer_pusher_camera_top_down,
        transpose=True,
        normalize=True,
    )

register_custom_envs()
