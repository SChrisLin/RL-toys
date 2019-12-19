# coding=UTF-8
'''
ppo在各种环境下的config
'''
class Config():
    def __init__(self, env_name):
        if env_name == 'Ant-v2':
            '''
            这一组超参数不算特别好
            '''
            self.env_name = 'Ant-v2'
            self.output_path = './save/ppo/{}/'.format(self.env_name)
            self.log_path = self.output_path + "log.txt"
            self.randomseed = 1 # random seed
            self.num_batches = 1000 # batch数量
            self.batch_size = 1000 # 每个batch多长
            self.max_ep_len = 1000 # 每条episode最大多长
            self.gamma = 0.95  # 折扣因子
            self.actor_lr = 0.001
            self.critic_lr = 0.003 
            self.actor_train_steps = 1
            self.critic_train_steps = 10
            self.num_units = 128
            self.record_frequency = 1
            self.lr_decay = True
            self.actor_lr_decay_rate = self.actor_lr / self.num_batches
            self.critic_lr_decay_rate = self.critic_lr / self.num_batches
            # 这个参数主要用在采集一个batch上，采集一个batch有两种方式一种是制定batch_size，一种是指定轨迹条数，默认为None表示指定batch_size
            self.num_episodes = None 
        if env_name == 'InvertedPendulum-v2':
            '''
            这一组超参数还行
            '''
            self.env_name = 'InvertedPendulum-v2'
            self.output_path = './save/ppo/{}/'.format(self.env_name)
            self.log_path = self.output_path + "log.txt"
            self.randomseed = 342  # random seed
            self.num_batches = 1000 # batch数量
            self.batch_size = 1000 # 每个batch多长
            self.max_ep_len = 1000 # 每条episode最大多长
            self.gamma = 1  # 折扣因子
            self.actor_lr = 0.001
            self.critic_lr = 0.002
            self.actor_train_steps = 1
            self.critic_train_steps = 10
            self.num_units = 32
            self.record_frequency = 1
            self.lr_decay = True
            self.actor_lr_decay_rate = self.actor_lr / self.num_batches
            self.critic_lr_decay_rate = self.critic_lr / self.num_batches
            self.num_episodes = None
        if env_name == 'Reacher-v2':
            '''
            待调参...
            '''
            self.env_name = 'Reacher-v2'
            self.output_path = './save/ppo/{}/'.format(self.env_name)
            self.log_path = self.output_path + "log.txt"
            self.randomseed = 1  # random seed
            self.num_batches = 10000 # batch数量
            self.batch_size = 200 # 每个batch多长
            self.max_ep_len = 32 # 每条episode最大多长
            self.gamma = 0.9  # 折扣因子
            self.actor_lr = 0.0001
            self.critic_lr = 0.0002 
            self.actor_train_steps = 10
            self.critic_train_steps = 10
            self.num_units = 100
            self.record_frequency = 5
            self.num_episodes = None



            
 

            

          

        

            
         