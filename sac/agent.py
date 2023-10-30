import os
import torch as T
import torch.nn.functional as F
import numpy as np


from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class Agent():
    #env_id: separate id for name of networks for separate storage files
    #env: for bounds
    def __init__(self, alpha, beta, input_dims, tau, env, env_id, gamma=0.99, n_actions=2, max_size=1000000,
                 layer1_size=256, layer2_size=256, batch_size=100, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size,input_dims,n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,  n_actions, max_action=env.action_space.high[0], name=env_id+'_actor' )
        self.critic_1 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_1')
        self.critic_2 = CriticNetwork(beta, input_dims, layer1_size, layer2_size, n_actions=n_actions,
                                      name=env_id+'_critic_2')
        self.value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id+'_value')
        self.target_value = ValueNetwork(beta, input_dims, layer1_size, layer2_size, name=env_id+'target_value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation],dtype=T.float32).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, reparameterize=False)

        return actions.cpu().detach().numpy()[0] #index 0 as it is multidim array

    def remember(self, state, action, reward, new_state,done):
        self.memory.store_transition(state,action,reward,new_state,done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()
        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
                value_state_dict[name] = tau*value_state_dict[name].clone() + (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('..saving models..')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('..loading models..')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        #retrieve batch of samples from buffer
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        reward = T.tensor(reward, dtype=T.float32).to(self.critic_1.device)
        done = T.tensor(done, dtype=T.bool).to(self.critic_1.device)
        state_ = T.tensor(new_state, dtype=T.float32).to(self.critic_1.device)
        state = T.tensor(state, dtype=T.float32).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float32).to(self.critic_1.device)

        #get value of current and future state
        value = self.value(state).view(-1) # scalar quantity of values: we can collapse the tensor along batch dimension
        value_ = self.target_value(state_).view(-1)
        value_[done] = 0 #value of terminal states in zero

        #retrieve an action
        actions, log_probs = self.actor.sample_normal(state, reparameterize=False) # retrieve an action
        #log_probs = log_probs.view(-1)
        #actions = actions.view(-1)

        #predict the q value for the current state and sampled action
        q1_new_policy = self.critic_1.forward(state, actions) # get action value from both critics
        q2_new_policy = self.critic_2.forward(state, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy) # choose the minimum to not become overconfident

        #update the value function with the difference between the current value and the predicted value of the critic
        self.value.optimizer.zero_grad()
        value_target = critic_value.view(-1) - log_probs.view(-1)
        # the value loss is difference (mse) between current and target value
        value_loss = 0.5 + F.mse_loss(value, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step() # update the value network

        #Actor loss
        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        log_probs = log_probs.view(-1)
        q1_new_policy = self.critic_1.forward(state,actions)
        q2_new_policy = self.critic_2.forward(state,actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        critic_value = critic_value.view(-1)

        actor_loss = log_probs - critic_value
        actor_loss = T.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        #Critic loss
        q_hat = self.scale * reward + self.gamma * value_ # value_ --> value of new states
        q1_old_policy = self.critic_1.forward(state, action).view(-1) # old policy: action from replay buffer instead of current action
        q2_old_policy = self.critic_2.forward(state, action).view(-1)
        critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.update_network_parameters()








