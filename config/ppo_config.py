# coding=UTF-8
'''
ppo在各种环境下的config
'''
class Config():
    def __init__(self, env_name):
        if env_name == 'Ant-v2':
            self.env_name = 'Ant-v2'
            self.output_path = './save/ppo/{}/'.format(self.env_name)
            self.log_path = self.output_path + "log.txt"
            self.randomseed = 1  # random seed
            self.num_batches = 500 # batch数量
            self.batch_size = 2000 # 每个batch多长
            self.max_ep_len = 2000 # 每条episode最大多长
            self.gamma = 0.95  # 折扣因子
            self.actor_lr = 0.001
            self.critic_lr = 0.003 
            self.actor_train_steps = 1
            self.critic_train_steps = 10
            self.num_units = 128
            # 这个参数主要用在采集一个batch上，采集一个batch有两种方式一种是制定batch_size，一种是指定轨迹条数，默认为None表示指定batch_size
            self.num_episodes = None 
        if env_name == 'InvertedPendulum-v2':
            self.env_name = 'InvertedPendulum-v2'
            self.output_path = './save/ppo/{}/'.format(self.env_name)
            self.log_path = self.output_path + "log.txt"
            self.randomseed = 1  # random seed
            self.num_batches = 1000 # batch数量
            self.batch_size = 500 # 每个batch多长
            self.max_ep_len = 500 # 每条episode最大多长
            self.gamma = 0.95  # 折扣因子
            self.actor_lr = 0.01
            self.critic_lr = 0.03 
            self.actor_train_steps = 1
            self.critic_train_steps = 10
            self.num_units = 64
            self.num_episodes = None


            
 

            

          

        

            
         