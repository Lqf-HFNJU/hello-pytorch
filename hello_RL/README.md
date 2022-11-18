# Reinforcement Learning

## critic
* #### 两种累积奖赏和的策略
  * T步累积奖赏
  * 做指数平滑
* #### 两种采样的方法
  * MC
    * base on data
    * 方差大：可能偏差很大
  * TD
    * 时序差分
    * 不精确：估测部分可能不准确
    * 目前一般用这个
  * 两者结合
    * 实际做DQN的时候，可以做一个Multi-Step
      * `Q(s(t), a(t)) = sum(r(t+1)~r(t+n)) + max{Q(s(t+n), a(t+n))}`
* #### V(s)
  * 价值函数
  * 看到某一个state的时候，的累积奖赏和
* #### Q(s,a)
  * Q函数
  * 看到某一个state的时候，强制做一个action的累积奖赏和

## DQN
* #### 一些计算tips
  1. Target Network
    * 网络的训练中要使用 `Q(s(t), a(t)) = rt + max{Q(s(t+1), a(t+1))}`来回归 `Q(s(t), a(t))`的值
    * 但是 Q 都是用网络来学出来的，如果每次都同时更新两个网络，会很难训练
    * 因此通常固定住产生`Q(s(t), a(t))`的网络，仅更新另一个网络，等到另一个网络迭代多次之后再将参数复制给这个 target 网络
  2. Exploration
    * 为了解决每次迭代后都会取 Q 中最好的那个 action，添加随机化的策略：
      * Epsilon Greedy
      * Boltzmann Exploration
  3. Replay Buffer
    * 取一个缓冲区缓存每次和环境的互动资料
    * 每个 buffer 里面的资料可能由迭代次数不同的 actor 产生，但是对 Q 的估计影响不大
  4. Double DQN
  5. Dual DQN
  6. ...

## policy gradient


## A3C
