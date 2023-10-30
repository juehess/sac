import pybullet_envs
import gym
import pybulletgym
import numpy as np
from agent import Agent
from utilities import plot_learning_curve

if __name__ == '__main__':
    env_id = 'InvertedPendulumSwingupPyBulletEnv-v0'

    env = gym.make(env_id)
    input_dims = env.observation_space.shape
    n_actions = env.action_space.shape[0]
    max_action = env.action_space.high

    agent = Agent(alpha=0.03, beta=0.0003, reward_scale=2, env_id=env_id, input_dims=env.observation_space.shape,
                  tau=0.005, env=env, batch_size=256, layer1_size=256, layer2_size=256, n_actions=env.action_space.shape[0])

    n_games = 250
    filename = env_id+'_'+str(n_games)+'games_scale'+str(agent.scale)+'.png'
    figure_file = 'tmp/plots/' + filename

    best_score = env.reward_range[0] #retrieve best reward from environment
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human') #to visualize performance of agents for us

    steps = 0
    for i in range(n_games):
        score = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
            env.render()
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode', i, 'score %1.f', score, '100 games average %1.f', avg_score, 'steps %d' % steps, env_id, 'scale', agent.scale)

        if not load_checkpoint:
            x = [i+1 for i in range(n_games)]
            plot_learning_curve(x, score_history, figure_file)