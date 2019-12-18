#%%
import os
import time
import argparse
import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from utils.general import get_logger

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='train', action='store_false')

args = parser.parse_args()
class Config():
    def __init__(self, env_name, use_baseline):
        if env_name == "CartPole-v0":
            self.env_name="CartPole-v0"
            self.record = True 
            baseline_str = 'baseline' if use_baseline else 'no_baseline'

            # output config
            self.output_path = "./save/pg/{}-{}/".format(self.env_name, baseline_str)
            self.model_output = self.output_path + "model.weights/"
            self.policy_model_output = self.model_output + "policy_model"
            self.baseline_model_output = self.model_output + "/baseline_model"
            
            # 待续...
            self.log_path     = self.output_path + "log.txt"
            self.plot_output  = self.output_path + "scores.png"
            self.record_path  = self.output_path 
            self.record_freq = 5
            self.summary_freq = 1
            
            # model and training config
            self.num_batches            = 200 # 训练多少个batch
            self.batch_size             = 1000 # 每个batch，最大步长
            self.max_ep_len             = 1000 # 每条路径的最大长度
            self.learning_rate          = 3e-2 # 学习率
            self.gamma                  = 1.0 # 折扣因子
            self.use_baseline           = use_baseline
            self.normalize_advantage    = False 

            # parameters for the policy and baseline models
            self.n_hidden_layers               = 2
            self.layer_units_size             = 16

            # show model result
            self.num_iter_in_show = 100000

            # since we start new episodes for each batch
            assert self.max_ep_len <= self.batch_size
            if self.max_ep_len < 0:
                self.max_ep_len = self.batch_size

        elif env_name == "HalfCheetah-v2":

            self.env_name="HalfCheetah-v2"
            self.record = True 
            baseline_str = 'baseline' if use_baseline else 'no_baseline'

            # output config
            self.output_path = "./save/pg/{}-{}/".format(self.env_name, baseline_str)
            self.model_output = self.output_path + "model.weights/"
            self.policy_model_output = self.model_output + "policy_model"
            self.baseline_model_output = self.model_output + "/baseline_model"
            
            # 待续...
            self.log_path     = self.output_path + "log.txt"
            self.plot_output  = self.output_path + "scores.png"
            self.record_path  = self.output_path 
            self.record_freq = 5
            self.summary_freq = 1
            
            # model and training config
            self.num_batches            = 1000 # 训练多少个batch
            self.batch_size             = 50000 # 每个batch，最大步长
            self.max_ep_len             = 1000 # 每条路径的最大长度
            self.learning_rate          = 3e-2 # 学习率
            self.gamma                  = 0.9 # 折扣因子
            self.use_baseline           = use_baseline
            self.normalize_advantage    = True 

            # parameters for the policy and baseline models
            self.n_hidden_layers               = 3
            self.layer_units_size             = 32

            # show model result
            self.num_iter_in_show = 100000

            # since we start new episodes for each batch
            assert self.max_ep_len <= self.batch_size
            if self.max_ep_len < 0:
                self.max_ep_len = self.batch_size


# pytorch
class Net(nn.Module):
    def __init__(self, input_size, 
                        output_size, 
                        n_hidden_layer, 
                        n_units):
        super(Net, self).__init__()
        self.n_hidden_layer = n_hidden_layer
        self.num_units = n_units
        self.num_out = output_size
        self.input_linear = nn.Linear(input_size, n_units)
        self.hidden_layer = nn.Linear(n_units, n_units)
        self.out_layer = nn.Linear(n_units, output_size)

    def forward(self, x):
        x = F.relu(self.input_linear(x))
        for _ in range(self.n_hidden_layer - 1):
            x = F.relu(self.hidden_layer(x))
        x = self.out_layer(x)
        return x


class PolicyGradient(object):
    def __init__(self, env, config, logger=None):
        '''
        初始化策略梯度类
        Args:
            env: open-ai中的环境或者自己写的环境，满足gym的接口
            config: class with hyperparameters
            logger: logger instance from logging module
        '''
        # 被保存模型的路径
        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)
        self.config = config

        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)

        self.env = env
        # 判断环境的动作空间是离散的还是连续的
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete) 
        # 对于CartPole-v0, observation_dim = 4
        # InvertedPendulum-v2, observation_dim.shape = 4
        self.observation_dim = self.env.observation_space.shape[0]
        # 离散情况是多少种动作，连续情况一个向量 
        # 对于CartPole-v0, action_dim.shape = 2
        # InvertedPendulum-v2, action_dim.shape = 1
        self.action_dim = self.env.action_space.n if self.discrete else self.env.action_space.shape[0]
        self.lr = self.config.learning_rate
        # build model
        self.build()
    
    def build(self):
        '''
        在build函数中构建策略网络，baseline网络
        ''' 
        # 得到策略网络模型和baseline网络模型
        if self.discrete:
            self.policy_net = Net(self.observation_dim, 
                                    self.action_dim, 
                                    self.config.n_hidden_layers,
                                    self.config.layer_units_size) 
            
        else:
            self.policy_net = Net(self.observation_dim, 
                                    self.action_dim, 
                                    self.config.n_hidden_layers,
                                    self.config.layer_units_size) 
        
        if self.config.use_baseline:
            self.baseline_net = Net(self.observation_dim, 
                                    1, 
                                    self.config.n_hidden_layers, 
                                    self.config.layer_units_size)
        # tensorboard 相关 定义一个writer
        self.writer = SummaryWriter(self.config.output_path)
    
    def train_policy_model(self, ob, action, advantage):
        '''
        训练策略网络
        args:
            ob: np.array shape = [None, obeservation_dim]
            action: np.array shape = [None, ？]
            advantage : np.array shape = [None, 1] 
        '''
        ob_tensor = torch.from_numpy(ob).float()
        action_label = torch.from_numpy(action).float() # 不是onehot编码，调用cal_logprob中会自动转化为onehot
        advantage_tensor = torch.from_numpy(advantage).float()

        optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        for _ in range(1):
            # 从numpy 来的数据是float64, 转化成float32
            # 梯度设置为0
            optimizer.zero_grad()
            # 前向传播
            outputs = self.policy_net(ob_tensor)
            logprpb = self.cal_logprob(outputs, action_label) # shape=[None, 1]
            loss = torch.mean(logprpb * advantage_tensor)
            # 反向传播
            loss.backward()
            # 进行优化
            optimizer.step()
    
    def train_baseline_model(self, ob, advantage):
        '''
        训练baseline网络
        args:
            ob: np.array shape = [None, obeservation_dim]
            advantage : np.array shape = [None, 1] 
        '''
        ob_tensor = torch.from_numpy(ob).float()
        advantage_tensor = torch.from_numpy(advantage).float()
        optimizer = optim.Adam(self.baseline_net.parameters(), lr=self.config.learning_rate)
        
        for _ in range(10):
            # 从numpy 来的数据是float64, 转化成float32
            # 梯度设置为0
            optimizer.zero_grad()
            # 前向传播
            outputs = self.baseline_net(ob_tensor)
            cri = nn.MSELoss()
            loss = cri(outputs.view(-1), advantage_tensor)
            # 反向传播
            loss.backward()
            # 进行优化
            optimizer.step()
    
    def cal_logprob(self, model_output, labels):
        '''
        输入：model_output:tensor shape=[batch_size, num_class], labels:tensor shape=[batch_size, 1]
        '''
        # 要把labels转化为稀疏形式，
        
        if self.discrete:
            labels = labels.long()
            logits = model_output
            criterion = nn.CrossEntropyLoss(reduction='none')
            logprob = criterion(logits, labels)
        else:
            loc = model_output # shape = [None, 1] or [None, k]
            logprob = self.cal_logprob_of_batch_normal(model_output, labels)
        return logprob
    
    def cal_logprob_of_batch_normal(self, model_output, real_action):
        '''
        待完善
        '''
        if model_output.shape[1] == 1:
            # 单变量正态分布
            logprob = -Normal(loc=model_output, scale=0.1).log_prob(real_action.view(-1, 1)).view(-1)
        elif model_output.shape[1] > 1:
            # # 多变量正态分布， 一个一个算速度太慢
            # len_model = model_output.shape[0]
            # logprob = torch.zeros(len_model)
            # for i, data in enumerate(zip(model_output, real_action)):
            #     mean, label = data[0], data[1]
            #     size_m = len(mean)
            #     logprob[i] = -MultivariateNormal(loc=mean, covariance_matrix=torch.eye(size_m)).log_prob(label)

            # 用单变量正态分布代替多变量正态分布，这里假设多变量正态分布的协方差是对角矩阵
            logprob = torch.sum(-Normal(loc=model_output, scale=0.1).log_prob(real_action), dim=1)
        return logprob

    def calculate_advantage(self, returns, observations):
        '''
        计算advantage
        '''
        adv = returns
        if self.config.use_baseline:
            # 一批观测的值函数, 多条路径的观测，也是很长的list, 长度为num_path*path_len
            b_n = torch.squeeze(self.baseline_net(torch.from_numpy(observations).float())).data.numpy()
            b_n = b_n * np.std(adv, axis=0) + np.mean(adv, axis=0)
            adv = adv - b_n  
        if self.config.normalize_advantage:
            adv_mean = np.mean(adv, axis=0)
            adv_std = np.std(adv, axis=0)
            adv = (adv - adv_mean) / (adv_std + 1e-7)  
        # 计算这一批路径的advantage，观测值函数-目标值函数
        return adv
    
    def init_averages(self):
        '''
        初始化回报的平均值
        '''
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.
    
    def sample_action(self, state):
        '''
        利用策略，根据状态得到动作
        args:
            state: np.array, shape = [None,]
        return:
            action: np.array, shape = 离散情况是[1] or 连续情况是[None,]
        '''
        if self.discrete:
            action_logits = self.policy_net(torch.from_numpy(state).float())
            action = Categorical(logits = action_logits).sample().view(-1).numpy()
        else:
            action_means = self.policy_net(torch.from_numpy(state).float())
            if action_means.shape[0] == 1:
                action = Normal(loc=action_means, scale=torch.tensor([1.0])).sample().numpy()
            else :
                action = MultivariateNormal(loc=action_means, covariance_matrix=torch.eye(action_means.shape[0])).sample().numpy()
        return action
    
    def sample_path(self, env, num_episodes = None):
        '''
        从环境中采集轨迹，输入环境，路径条数
        采集一个batch，多条路径构成一个batch, 一个batch的长度固定
        '''
        episode = 0
        episode_rewards = []
        paths = []
        t = 0 # 记录全局长度，全局长度不超过
        while (num_episodes or t < self.config.batch_size): # batch_size条路径
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0
            for step in range(self.config.max_ep_len): # max_ep_len：每天路径最大长度
                states.append(state)
                # 根据模型得到动作，action是一个数组
                action = self.sample_action(states[-1])
                # 把action数组 [1] 变成数字 1
                if self.discrete:
                    action = action[0]
                # state是array,reward是一个数
                state, reward, done, _ = env.step(action)
                actions.append(action)
                rewards.append(reward)
                # 每条路径的回报
                episode_reward += reward
                t += 1
                # 如果一条路径结束，或者已经到了一条路径的最大步长
                if (done or step == self.config.max_ep_len-1):
                    episode_rewards.append(episode_reward)
                    break

                # 如果这个batch的长度已经到了batch_size，则已经构成一个batch
                # 此处用于中断一条路径继续执行下去，跳出循环后会因为t == self.config.batch_size 而跳出大循环
                if (not num_episodes) and t == self.config.batch_size:
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
    
    def get_returns(self, paths):
        '''
        根据一条或者多条轨迹时间步上的累计回报：Gt
        '''
        all_returns = []
        for path in paths:
            # 一条路径中一组汇报rt
            rewards = path["reward"]
            max_step = len(rewards)
            # 计算Gt,也是一组
            path_returns = [np.sum(np.power(self.config.gamma, np.arange(max_step - t)) * rewards[t:]) for t in range(max_step)]
            all_returns.append(path_returns)
        # 拼接回报, 横着拼接，变成很长的list
        returns = np.concatenate(all_returns)
        return returns
    
    def update_averages(self, rewards, scores_eval):
        '''
        计算一组rewards的平均值，最大值，标准差
        Args:
            rewards: deque
            scores_eval: list
        '''
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))
        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]
    
    def train(self):
        '''
        开始训练
        '''
        self.init_averages()
        scores_eval = [] 
        for t in range(self.config.num_batches):
            # 采样轨迹，得到轨迹和总回报
            paths, total_rewards = self.sample_path(self.env) 
            scores_eval = scores_eval + total_rewards
            # 得到所有路径的观测值，动作，以及回报
            observations = np.concatenate([path["observation"] for path in paths])
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])

            # compute Q-val estimates (discounted future returns) for each time step
            # 根据路径(一条)，得到一组Gt, 并且计算At
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            # 根据Gt,和观测更新baseline网络
            if self.config.use_baseline:
                q_n_mean = np.mean(returns, axis=0)
                q_n_std = np.std(returns, axis=0)
                q_n = (returns - q_n_mean) / (q_n_std + 1e-7)
                self.train_baseline_model(observations, q_n)
                
            # 根据采集到的一条路径，训练策略网络
            self.train_policy_model(observations, actions, advantages)

            # 添加tensorbord 记录
            if (t % self.config.summary_freq == 0):
                self.update_averages(total_rewards, scores_eval)
                self.record_summary(t)

            # 一条路径或者一批路径的平均回报
            avg_reward = np.mean(total_rewards)
            # 一条路径或一批路径的方差
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward, sigma_reward)

            scores_eval
            self.logger.info(msg)
        self.logger.info("- Training done.")
        # 保存模型
        self.save()
        self.writer.close()
        
    def record_summary(self, t):
        '''
        tensorboard 添加记录
        '''
        self.writer.add_scalar('avg_reward', self.avg_reward, t)
        self.writer.add_scalar('max_reward', self.max_reward, t)
        self.writer.add_scalar('std_reward', self.std_reward, t)
        self.writer.add_scalar('eval_reward', self.eval_reward, t)

    def save(self):
        '''
        训练完成后保存模型
        '''
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)
        torch.save(self.policy_net.state_dict(), self.config.policy_model_output)
        if self.config.use_baseline:
            torch.save(self.baseline_net.state_dict(), self.config.baseline_model_output)
    
    def load(self):
        '''
        导入模型
        '''
        self.policy_net.load_state_dict(torch.load(self.config.policy_model_output))
        self.policy_net.eval()
        if self.config.use_baseline:
            self.baseline_net.load_state_dict(torch.load(self.config.baseline_model_output))
            self.baseline_net.eval()
        
    def run(self):
        self.train()
    
if __name__ == '__main__':
    # 这里不加baseline, 也不归一化回报，就是纯粹的策略梯度方法
    # 减去baseline效果会更好
    config = Config('CartPole-v0', False) 
    # config = Config('HalfCheetah-v2', True) 
    env = gym.make(config.env_name)
    model = PolicyGradient(env, config)
    if args.train == True:
        # train model
        model.run()
    else:
        model.load()
        for i in range(20):
            ob = env.reset()
            for t in range(config.num_iter_in_show):
                env.render()
                action = model.sample_action(ob)      
                ob, reward, done, info = env.step(action[0])
                if done:
                    print('episode finished after{}timesteps'.format(t+1))
                    break
        env.close()    