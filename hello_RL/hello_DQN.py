"""
DQN
    用神经网络去学习 Q 解放了 Q 一系列定义、计算参数可能过多而难算的问题
    这个 reward 至关重要！！！

    gym 库的 CartPole使用：
        env = gym.make("CartPole-v0")
        observation = env.reset()
        for _ in range(1000):
            env.render()  # 打开环境窗口
            action = env.action_space.sample()  # 从行为空间随机挑选动作
            observation, reward, done, info = env.step(action)  # 执行该action的返回

            if done:
                observation = env.reset()
        env.close()  # 关闭环境
"""
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import gym

""" 超参数 """
n_epochs = 400
batch_size = 32
lr = 0.01
epsilon = 0.8  # epsilon greedy 策略
gamma = 0.9  # 指数平滑的比例
hidden_size = 64
target_replace_iter = 100  # target 网络的更新频率
memory_capacity = 2000  # 缓冲区容量

device = 'cuda' if torch.cuda.is_available() else 'cpu'

""" 游戏设定 """
env = gym.make("CartPole-v0").unwrapped  # 使用gym库中的环境：CartPole，且打开封装
N_ACTIONS = env.action_space.n  # 杆子动作个数 (2个)
N_STATES = env.observation_space.shape[0]  # 杆子状态个数 (4个)

""" 
网络
    state -> Q
    一个更新网络 一个target 两个网络的结构一样
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)  # 将网络初始化参数设置为方差为0.1的正态分布
        self.out = nn.Linear(hidden_size, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0  # 迭代多少次了
        self.memory_counter = 0  # buffer 存储次数
        self.memory = torch.FloatTensor(memory_capacity, N_STATES * 2 + 2).to(device)  # 容量*(s(t+1),a(t),r(t),s(t))
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def choose_action(self, x):
        """ 选择 action 的策略：Epsilon Greedy """
        x = torch.unsqueeze(torch.FloatTensor(x).to(device), 0)
        if np.random.uniform() < epsilon:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0]
        else:
            action = np.random.randint(0, N_ACTIONS)
        return action

    def store_transition(self, s, a, r, s_):
        """ 实现 buffer 的存储管理 """
        transition = torch.FloatTensor(np.hstack((s, [a, r], s_))).to(device)  # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        index = self.memory_counter % memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target network 参数更新
        if self.learn_step_counter % target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(memory_capacity, batch_size)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]
        # b_s -> (32, 4)
        b_s = b_memory[:, :N_STATES]
        # b_a -> (32, 1) (之所以为LongTensor类型，是为了方便后面torch.gather的使用)
        b_a = b_memory[:, N_STATES:N_STATES + 1].long()
        # b_r -> (32, 1)
        b_r = b_memory[:, N_STATES + 1:N_STATES + 2]
        # b_s_ -> (32, 4)
        b_s_ = b_memory[:, -N_STATES:]

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        # gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + gamma * q_next.max(1)[0].view(batch_size, 1)
        loss = self.loss(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def test(model):
    s = env.reset()[0]
    episode_reward_sum = 0
    while True:
        env.render()
        a = model.choose_action(s)
        s_, r, done, info, dic = env.step(a)

        model.store_transition(s, a, r, s_)
        episode_reward_sum += r

        s = s_
        if done:
            print('test----->reward_sum: %s' % (episode_reward_sum))
            break


if __name__ == '__main__':
    model = DQN()
    for i in range(n_epochs):
        print('<<<<<<<<<Episode: %s' % i)
        s = env.reset()[0]  # 重置环境
        episode_reward_sum = 0

        while True:
            env.render()  # 显示实验动画
            a = model.choose_action(s)
            s_, r, done, info, _ = env.step(a)

            # 调整奖赏可以更好训练
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
            new_r = r1 + r2

            model.store_transition(s, a, new_r, s_)
            episode_reward_sum += new_r

            s = s_  # 更新状态

            if model.memory_counter > memory_capacity:
                model.learn()

            if done:
                print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
                break

    test(model)
    env.close()
