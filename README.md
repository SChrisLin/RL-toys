# Reinforcement Learning Toys
一些常见强化学习算法简单代码实现。  
## 运行要求
- Python 3.X
- 安装gym, mujoco [详见:gym](https://github.com/openai/gym)
- 安装pytorch, [详见:pytorch](https://pytorch.org/)

## Policy Gradient 
- 训练策略网络
    ```shell
    $ cd RL-toys
    $ python pg.py --train
    ```
- 训练完成后查看训练过程中每个batch回报值变化
    ```shell
    $ tensorboard --logdir=pg_save
    ```
    可在网页中看到每个batch的平均回报、最大回报等数据：  

    ![](resource_md/img/pg_cart_tensorboard.png)  

- 测试训练好的模型
    ```shell
    $ python pg.py --test
    ```

