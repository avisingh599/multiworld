import gym
from multiworld.envs.mujoco import register_goal_example_envs
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np

def sawyer_pusher_camera_overview(camera):
    camera.distance = 0.7
    camera.lookat[0] = 0
    camera.lookat[1] = 0.85
    camera.lookat[2] = 0.45
    camera.elevation = -155
    camera.azimuth = 90
    camera.trackbodyid = -1

def sawyer_door_camera_overview(camera):
    camera.distance = 2.0
    camera.lookat[0] = 0
    camera.lookat[1] = 0.45
    camera.lookat[2] = 0.15
    camera.elevation = -130
    camera.azimuth = -30
    camera.trackbodyid = -1

def main():
    register_goal_example_envs()
    #task = 'Image84SawyerPushSidewaysEnv-v0'
    #task = 'SawyerDoorHookResetFreeEnvImage48-v1'
    task = 'Image84SawyerDoorPullHookEnv-v0'
    env = gym.make(task)
    #env = gym.make()
    #env = gym.make('SawyerDoorHookResetFreeEnv-v1')

    #env = gym.make('SawyerDoorHookEnv-v0')

    #ob = env.reset()
    #import IPython; IPython.embed()
    

    #add goal marker
    # qpos = env.data.qpos.flat.copy()
    # qvel = env.data.qvel.flat.copy()
    # qpos[15:18] = np.asarray([-0.15, 0.6, 0.05])
    # env.set_state(qpos, qvel)

    # img = env.sim.render(500, 500)
    # plt.imshow(img)
    # plt.show()

    #overview_viewer = mujoco_py.MjRenderContextOffscreen(env.sim, device_id=-1)
    #sawyer_door_camera_overview(overview_viewer.cam)
    #sawyer_pusher_camera_overview(overview_viewer.cam)
    
    #overview_viewer.render(500, 500, camera_id=None)
    #img = overview_viewer.read_pixels(500, 500, None)
    #plt.imshow(img)
    #plt.show()

    # import IPython; IPython.embed()

    

if __name__ == "__main__":
    main()
