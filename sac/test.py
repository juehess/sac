#import gym
#import pybullet_envs
import pybulletgym
import gym
env_name = 'InvertedPendulumSwingupPyBulletEnv-v0'
env = gym.make(env_name)

env.render()
env.reset()

for i in range(10000):
    obs, rewards, done, _ = env.step(env.action_space.sample())