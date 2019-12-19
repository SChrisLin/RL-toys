# coding=UTF-8
'''
Proximal Policy Optimization (PPO)
----------------------------
A simple version of Proximal Policy Optimization (PPO) using single thread.
PPO is a family of first-order methods that use a few other tricks to keep new policies close to old.
PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

Reference
---------
Proximal Policy Optimization Algorithms, Schulman et al. 2017
High Dimensional Continuous Control Using Generalized Advantage Estimation, Schulman et al. 2016
Emergence of Locomotion Behaviours in Rich Environments, Heess et al. 2017
'''
import argparse
import os
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

from config import ppo_config
from utils.general import get_logger

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class Actor(nn.Module):
    '''
    policy 网络
    '''
    def __init__(self, input_size, 
                       output_size, 
                       n_units):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, n_units)
        self.fc2 = nn.Linear(n_units, output_size)
        self.fc3 = nn.Linear(n_units, output_size)
    
    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        a = torch.tanh(self.fc2(l1))
        mu =  2 * a 
        sigma = F.softplus(self.fc3(l1))
        return [mu, sigma]

class Critic(nn.Module):
    '''
    critic 网络
    '''
    def __init__(self, input_size, 
                       output_size, 
                       n_units):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.n_units = n_units
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, n_units)
        self.fc2 = nn.Linear(n_units, 1)
    
    def forward(self, x):
        l1 = F.relu(self.fc1(x))
        v = self.fc2(l1)
        return v
# 命令端输入解析
parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--env', dest='env')
args = parser.parse_args()
TRAIN = args.train
# clip方法比原始版本的ppo更好
METHOD = dict(name='clip', epsilon=0.2) 

class PPO(object):
    '''
    PPO class
    '''
    def __init__(self, env, config):
        self.config = config
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
        # 定义critic网络，critic
        self.critic = Critic(S_DIM, A_DIM, self.config.num_units)
        # 定义策略网络
        self.actor = Actor(S_DIM, A_DIM, self.config.num_units)
        self.actor_old = Actor(S_DIM, A_DIM, self.config.num_units)       
        # 得到环境
        self.env = env 
        # logger
        self.logger = get_logger(self.config.log_path)
        # tensorboard 相关 定义一个writer
        self.writer = SummaryWriter(self.config.output_path)
        
    def train_actor_model(self, ob, action, advantage):
        '''
        训练策略
        '''
        ob_tensor = torch.from_numpy(ob).float()
        action_label = torch.from_numpy(action).float() 
        advantage_tensor = torch.from_numpy(advantage).float()
        optimizer = optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        for _ in range(self.config.actor_train_steps):
            # 梯度设置为0
            optimizer.zero_grad()
            mu, sigma = self.actor(ob_tensor)
            pi = torch.distributions.Normal(mu, sigma)
            mu_old, sigma_old = self.actor_old(ob_tensor)
            pi_old = torch.distributions.Normal(mu_old, sigma_old)

            log_prob_pi = torch.sum(pi.log_prob(action_label), dim=1) # (batch_size,)
            log_prob_pi_old = torch.sum(pi_old.log_prob(action_label), dim=1) # (batch_size,)
            ratio = torch.exp(log_prob_pi - log_prob_pi_old) # (batch_size,)
            surr = ratio * advantage_tensor # (batch_size,)
            
            aloss = -torch.mean(torch.min(surr, torch.clamp(ratio, 1.0-METHOD['epsilon'], 1.0+METHOD['epsilon']) * advantage_tensor))
            # 反向传播
            aloss.backward()
            # 进行优化
            optimizer.step()
    
    def train_critic_model(self, ob, advantage):
        '''
        训练critic网络
        '''
        ob_tensor = torch.from_numpy(ob).float()
        advantage_tensor = torch.from_numpy(advantage).float()
        optimizer = optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        
        for _ in range(self.config.critic_train_steps):
            # 从numpy 来的数据是float64, 转化成float32
            # 梯度设置为0
            optimizer.zero_grad()
            # 前向传播
            v = self.critic(ob_tensor)
            cri = nn.MSELoss()
            loss = cri(v.view(-1), advantage_tensor) # v.view(-1) 必须有
            # 反向传播
            loss.backward()
            # 进行优化
            optimizer.step()
    
    def update_old_pi(self):
        '''
        将actor模型的权重赋值给actor_old
        '''
        self.actor_old.load_state_dict(self.actor.state_dict())
    
    def init_averages(self):
        """
        初始化回报的平均值
        """
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.
        
    def update_averages(self, rewards, scores_eval):
        """
        计算一组rewards的平均值，最大值，标准差
    
        Args:
                rewards: deque
        """
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]
        
    def record_summary(self, t):
        """
        Add summary to tfboard
        """
        self.writer.add_scalar('avg_reward', self.avg_reward, t)
        self.writer.add_scalar('max_reward', self.max_reward, t)
        self.writer.add_scalar('std_reward', self.std_reward, t)
        self.writer.add_scalar('eval_reward', self.eval_reward, t)

    def calculate_advantage(self, returns, observations):
        '''
        Calculate advantage
        :param returns: cumulative reward
        :param observations: observations 一系列的ob 
        :return: advantage , 一个一维向量
        '''
        adv = returns
        # 一批观测的值函数, 多条路径的观测，也是很长的list, 长度为num_path*path_len
        obs = torch.from_numpy(observations).float()
        qvalue = self.critic(obs)
        b_n = torch.squeeze(qvalue).data.numpy()
        # b_n = b_n * np.std(adv, axis=0) + np.mean(adv, axis=0)
        adv = adv - b_n  
        # 计算这一批路径的advantage，观测值函数-目标值函数
        return adv

    def train(self):
        """
        开始训练
        """
        self.logger.info("- Training begin.")
        best_avg_reward = -10000
        scores_eval = [] 
        for t in range(self.config.num_batches):
            # 采样轨迹，得到轨迹和总回报
            paths, total_rewards = self.sample_path(self.env) 
            scores_eval = scores_eval + total_rewards
            # 得到所有路径的观测值，动作，以及回报
            observations, actions, rewards = self.get_all_SAR(paths)
            # 计算advantage
            advantages = self.calculate_advantage(rewards, observations)
            # 更新old actor网络
            self.update_old_pi()
            # update actor
            self.train_actor_model(observations, actions, advantages)
            # update critic
            self.train_critic_model(observations, advantages)
            # 一条路径或者一批路径的平均回报
            avg_reward = np.mean(total_rewards)
            # 一条路径或一批路径的方差
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "{:d}/{:d}:Average reward: {:04.2f} +/- {:04.2f}".format(t+1, self.config.num_batches, avg_reward, sigma_reward)

            # learning rate decay
            if self.config.lr_decay == True:
                self.config.critic_lr -= self.config.critic_lr_decay_rate
                self.config.actor_lr -= self.config.actor_lr_decay_rate
            # 添加tensorbord 记录
            if ((t+1) % self.config.record_frequency == 0):
                self.update_averages(total_rewards, scores_eval)
                self.record_summary((t+1)*self.config.batch_size)

            # print(msg)
            # 保存最佳模型
            if avg_reward >= best_avg_reward:
                self.save()
                best_avg_reward = avg_reward
                self.logger.info('========best model saved========')
            self.logger.info(msg)
        self.logger.info("- Training done.")
        
        
    def sample_action(self, state, use=False):
        '''
        利用策略，根据状态得到动作
        args:
            state: np.array, shape = [None,]
        return:
            action: np.array, shape = [None,]
        '''
        s = state[np.newaxis,:] # 加一个维度 [[1,2,3]]
        action_means, sigma = self.actor(torch.from_numpy(s).float())
        if use == False:
            # 具有随机性，用在train时期
            action = torch.squeeze(Normal(loc=action_means, scale=sigma).sample(), axis=0).numpy()
        else:
            # 确定性策略，每次都选最好的，用在test期间
            action = torch.squeeze(action_means, axis=0).data.numpy()
        return action
    
    def sample_path(self, env):
        '''
        从环境中采集轨迹，输入环境，路径条数
        采集一个batch，多条路径构成一个batch, 一个batch的长度固定
        '''
        batch_size = self.config.batch_size
        max_ep_len = self.config.max_ep_len
        num_episodes = self.config.num_episodes
        episode = 0
        episode_rewards = []
        paths = []
        t = 0 # 记录全局长度，全局长度不超过
        while (num_episodes or t < batch_size): # batch_size条路径
            state = env.reset() # 环境返回的是一个一维(特征向量)或者二维(图像)的数组
            states, actions, rewards = [], [], []
            episode_reward = 0
            for step in range(max_ep_len): # max_ep_len：每条路径的最大长度
                states.append(state)
                # 根据模型得到动作，action是一个数组
                action = self.sample_action(state)
                # state是array,reward是一个数
                state, reward, done, _ = env.step(action)
                # env.render()
                actions.append(action)
                rewards.append(reward)
                # 每条路径的回报
                episode_reward += reward
                t += 1
                # 如果一条路径结束，或者已经到了一条路径的最大步长
                if (done or step == max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break
                # 如果这个batch的长度已经到了batch_size，则已经构成一个batch
                # 此处用于中断一条路径继续执行下去，跳出循环后会因为t == self.config.batch_size 而跳出大循环
                if (not num_episodes) and t == batch_size:
                    break
            path = {"observation" : np.array(states),
                    "reward" : np.array(rewards),
                    "action" : np.array(actions)}
            paths.append(path)
            episode += 1
            # 如果已经采集了num_episodes条路径
            if num_episodes and episode >= num_episodes:
                break
        return paths, episode_rewards

    def get_all_SAR(self, paths):
        """
        根据多条paths，得到一连串states,actions,returns
        """
        all_returns = []
        for path in paths:
            # calculate Gt of a path
            # 使用critic评估
            s_ = path['observation'][-1]
            buffer_r = path['reward']
            v_s_ = self.get_v(s_) # 得到最后状态的V(s)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + self.config.gamma * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()
            buffer_r = []
            # store Gt   
            all_returns.append(discounted_r)

            # # 一条轨迹的最后一步，不使用critic评估
            # # 一条路径中一组汇报rt
            # rewards = path["reward"]
            # max_step = len(rewards)
            # # 计算Gt,也是一组
            # path_returns = [np.sum(np.power(self.config.gamma, np.arange(max_step - t)) * rewards[t:]) for t in range(max_step)]
            # all_returns.append(path_returns)

        # 拼接回报, 横着拼接，变成很长的list
        states = np.concatenate([path['observation'] for path in paths])
        actions = np.concatenate([path['action'] for path in paths])
        returns = np.concatenate(all_returns)

        return states,actions,returns

    def get_v(self, s):
        '''
        值函数
        args:
            s : state
        return:
            a float value
        '''
        state = torch.from_numpy(s).float()
        return torch.squeeze(self.critic(state)).data.numpy()
    
    def save(self):
        '''
        save model
        '''
        if not os.path.exists(self.config.output_path):
            os.makedirs(self.config.output_path)
        torch.save(self.actor.state_dict(), os.path.join(self.config.output_path, 'actor_model.pkl'))
        torch.save(self.actor_old.state_dict(), os.path.join(self.config.output_path, 'actor_old_model.pkl'))
        torch.save(self.critic.state_dict(), os.path.join(self.config.output_path, 'critic_model.pkl'))
    
    def load(self):
        '''
        load model
        '''
        self.actor.load_state_dict(torch.load(os.path.join(self.config.output_path, 'actor_model.pkl')))
        self.actor_old.load_state_dict(torch.load(os.path.join(self.config.output_path, 'actor_old_model.pkl')))
        self.critic.load_state_dict(torch.load(os.path.join(self.config.output_path, 'critic_model.pkl')))
        self.actor.eval()
        self.actor_old.eval()
        self.critic.eval()
        


if __name__ == '__main__':
    env_name = args.env
    assert env_name in ['InvertedPendulum-v2', 'Ant-v2', 'Pendulum-v0'], '没有输入正确的环境名称'
         
    # env = gym.make('InvertedPendulum-v2')
    # env = gym.make('Pendulum-v0')
    # env = gym.make('MountainCarContinuous-v0')
    # env = gym.make('Ant-v2')
    # env = gym.make('Reacher-v2')
    cfg = ppo_config.Config(env_name)
    env = gym.make(cfg.env_name)

    S_DIM, A_DIM = env.observation_space.shape[0], env.action_space.shape[0]   # state dimension, action dimension
   
    # 设置随机数种子，保证可复现
    env.seed(cfg.randomseed)
    np.random.seed(cfg.randomseed)
    torch.random.manual_seed(cfg.randomseed)

    ppo = PPO(env, cfg)
       
    if TRAIN:
        ppo.train()
    else:
        # 展现模型效果
        ppo.load()
        while True:
            s = env.reset()
            for i in range(5000):
                s, r, done, _ = env.step(ppo.sample_action(s, use=False))
                env.render()
                if done:
                    print('done')
                    break
