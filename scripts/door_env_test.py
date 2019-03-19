from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door_hook import SawyerDoorHookEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

from softlearning.misc.utils import save_video
import numpy as np
import imageio

import mujoco_py

def sawyer_door_camera_overview(camera):
    camera.distance = 2.0
    camera.lookat[0] = 0
    camera.lookat[1] = 0.45
    camera.lookat[2] = 0.15
    camera.elevation = -130
    camera.azimuth = -30
    camera.trackbodyid = -1

def main():
    
    # import IPython; IPython.embed()
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v0, \
        sawyer_pusher_camera_upright_v2, sawyer_pusher_camera_upright_v3, \
        sawyer_door_env_camera_v0


    goal_vec = {
        'state_desired_goal': np.asarray([0.05, 0.50, 0.15, +0.60])
    }

    env = ImageEnv(
            SawyerDoorHookEnv(
                fix_goal=True,
                fixed_goal=goal_vec['state_desired_goal'],
                indicator_threshold=(0.1, 0.03),
                reward_type='angle_success',
                hand_low=(-0.1, 0.45, 0.1),
                hand_high=(0.05, 0.65, .25),
                min_angle=-0.50,
                max_angle=.00,
                reset_free=False,
                ),
            init_camera=sawyer_door_env_camera_v0,
            #normalize=True,
            presampled_goals={'state_desired_goal': np.expand_dims(goal_vec['state_desired_goal'], axis=0),
                              'image_desired_goal': np.zeros((1, 21168))},
            non_presampled_goal_img_is_garbage=True,
            )

    env_flat = FlatGoalEnv(env, obs_keys=['state_observation', 'image_observation'])

    for i in range(10):
        env_flat.reset()
        ob, rew, done, info = env_flat.step(np.asarray([0.,0.,0.]))
        goal_vec['state_desired_goal'][:3] += np.random.uniform(low=-0.01, high=0.01, size=(3,))
        goal_vec['state_desired_goal'][3] += np.random.uniform(low=-0.01, high=0.01)
        
        env_flat.set_to_goal_pos(goal_vec['state_desired_goal'][:3])
        env_flat.set_to_goal_angle(goal_vec['state_desired_goal'][3])

        ob, rew, done, info = env_flat.step(np.asarray([0.,0.,0.]))
        print(ob[:4])
        imageio.imwrite('/root/ray_results/video_test/door_expert_image_{}.png'.format(i), ob[4:].reshape(84, 84, 3).astype(np.uint8))

    overview_viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
    sawyer_door_camera_overview(overview_viewer.cam)

    ob_list = []
    env_flat.reset()
    for i in range(100):
        
        if i < 50:
            act = [0.1, 0.5, -0.1]
        elif i < 85:
            act = [0.0, -0.2, 0.0]
        elif i < 100:
            act = [0.0, -0.0, 0.0]
        
        ob, rew, done, info = env_flat.step(np.asarray(act))
        overview_viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
        sawyer_door_camera_overview(overview_viewer.cam)
        overview_viewer.render(500, 500, camera_id=None)
        img = overview_viewer.read_pixels(500, 500, None)

        print(ob[:4])
        print(rew)
        #ob_list.append(ob[4:].astype(np.uint8))
        ob_list.append(img)

    ob_list = np.asarray(ob_list)
    ob_list = ob_list.reshape((-1, 500, 500, 3))

    save_video(ob_list, '/root/ray_results/video_test/SawyerDoor.avi')



if __name__ == "__main__":
    main()
