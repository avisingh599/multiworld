from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import \
    SawyerPushAndReachXYEnv, SawyerPushAndReachXYZEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_door import SawyerDoorEnv
from multiworld.core.image_env import ImageEnv
from multiworld.core.flat_goal_env import FlatGoalEnv

from softlearning.misc.utils import save_video
import numpy as np
import imageio


def main():
    
    # import IPython; IPython.embed()
    from multiworld.envs.mujoco.cameras import sawyer_pusher_camera_upright_v0, \
        sawyer_pusher_camera_upright_v2, sawyer_pusher_camera_upright_v3, \
        sawyer_door_env_camera_v0

#     goal_vec = {
#         #'state_desired_goal': np.asarray([0.0, 0.7, 0.02, 0.0, 0.8])
#         'state_desired_goal': np.asarray([0.0, 0.6, 0.02, -0.15, 0.6])
#     }
    
#     env = ImageEnv(
#             SawyerPushAndReachXYEnv(
#                 fix_goal=True,
#                 reward_type='puck_distance',
#                 #fixed_goal=(0.15, 0.6, 0.02, -0.15, 0.6),
#                 fixed_goal=goal_vec['state_desired_goal'],
#                 puck_radius=.05,
# #                puck_low=(-.1, .45),
# #                puck_high=(.1, .65),
# #                clamp_puck_on_step=True,
#                 indicator_threshold=0.03,
#                 ),
#             init_camera=sawyer_pusher_camera_upright_v3,
#             normalize=True,
#         )

    # for i in range(10):
    #     env_flat.reset()
    #     ob, rew, done, info = env_flat.step(np.asarray([0.,0.]))
    #     #print(info['puck_success'])
    #     goal_vec['state_desired_goal'][:2] += np.random.uniform(low=-0.01, high=0.01, size=(2,))
    #     goal_vec['state_desired_goal'][-2:] += np.random.uniform(low=-0.01, high=0.01, size=(2,))
        
    #     env_flat.set_to_goal(goal_vec)
    #     ob, rew, done, info = env_flat.step(np.asarray([0.,0.]))
    #     imageio.imwrite('/root/ray_results/video_test/door_expert_image_{}.png'.format(i), ob.reshape(84, 84, 3))
    #     print(info['puck_success'])
    #     #env_flat._set_puck_xy([0.0, 0.8])
    #     #env_flat._set_puck_xy([0.0, 0.8])


    goal_vec = {
        'state_desired_goal': np.asarray([0.2, 0.50, 0.12, -0.30])
    }

    env = ImageEnv(
            SawyerDoorEnv(
                fix_goal=True,
                fixed_goal=goal_vec['state_desired_goal'],
                indicator_threshold=(0.1, 0.03),
                reward_type='angle_success',
                hand_low=(-0.0, 0.45, 0.1),
                hand_high=(0.25, 0.65, .25),
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
    # env_flat.reset()
    # ob1, rew, done, info = env_flat.step(np.asarray([0.,0.]))

    # imageio.imwrite('/root/ray_results/video_test/door_initial_image.png', ob1.reshape(84, 84, 3))


    for i in range(10):
        env_flat.reset()
        ob, rew, done, info = env_flat.step(np.asarray([0.,0.]))
        goal_vec['state_desired_goal'][:3] += np.random.uniform(low=-0.01, high=0.01, size=(3,))
        goal_vec['state_desired_goal'][3] += np.random.uniform(low=-0.01, high=0.01)
        
        env_flat.set_to_goal_pos(goal_vec['state_desired_goal'][:3])
        env_flat.set_to_goal_angle(goal_vec['state_desired_goal'][3])

        ob, rew, done, info = env_flat.step(np.asarray([0.,0.]))
        print(ob[:4])
        imageio.imwrite('/root/ray_results/video_test/door_expert_image_{}.png'.format(i), ob[4:].reshape(84, 84, 3).astype(np.uint8))

    ob_list = []
    env_flat.reset()
    for i in range(100):
        #ob, rew, done, info = env_flat.step(np.asarray([0.0, 0.5]))
        # ob, rew, done, info = env_flat.step(np.random.uniform(low=-1.0, high=1.0, size=(2,)))
        #ob, rew, done, info = env_flat.step(np.random.uniform(low=0.0, high=1.0, size=(2,)))
        if i < 50:
            act = [1.0, 0.5]
        elif i < 100:
            act = [0.0, -0.5]
        
        ob, rew, done, info = env_flat.step(np.asarray(act))

        print(ob[:4])
        print(rew)
        ob_list.append(ob[4:].astype(np.uint8))

    ob_list = np.asarray(ob_list)
    ob_list = ob_list.reshape((-1, 84, 84, 3))

    save_video(ob_list, '/root/ray_results/video_test/SawyerDoor.avi')



if __name__ == "__main__":
    main()
