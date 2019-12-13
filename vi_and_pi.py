'''
基于动态规划的策略评估与策略迭代算法，还包括值迭代算法。
假设环境中，状态离散有限、动作离散有限、转移矩阵已知
变量解释：
	P: nested dictionary
		对于每一个状态s和动作a, P[s][a] 是一个元组，其中值为(probability, nextstate, reward, terminal)
			- probability: float
				在状态s下执行动作a转移到下一个状态的概率
			- nextstate: int
				下一个状态
			- reward: int
				转移到下一个状态的回报
			- terminal: bool
			  	下一个状态是否为中止状态
	nS: int
		离散状态的个数
	nA: int
		离散动作的个数
	gamma: float
		折扣因子，范围[0, 1)
本代码为stanford-CS234 2019 assignment1.
'''
import numpy as np
import gym
import time
from utils.lake_envs import *
import argparse
np.set_printoptions(precision=3)

parser = argparse.ArgumentParser(description='Policy iteration or value iteration.')
parser.add_argument('--policy', dest='policy', action='store_true', default=True)
parser.add_argument('--value',  dest='policy', action='store_false')

args = parser.parse_args()

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	'''
	策略评估
	args:
		P, nS, nA, gamma:
			defined at beginning of file
		policy: np.array[nS]
			The policy to evaluate. Maps states to actions.
		tol: float
			Terminate policy evaluation when
				max |value_function(s) - prev_value_function(s)| < tol
	returns
		value_function: np.ndarray[nS]
			The value function of the given policy, where value_function[s] is
			the value of state s
	'''
	value_function = np.zeros(nS)
	value_function_old = np.ones(nS) * np.inf
	Reward_pi = np.zeros(nS)
	for state in range(nS):
		# 在状态s执行动作a的回报, a是由固定策略policy决定
		action = policy[state] 
		# 策略固定只有一项，转移概率 * 回报
		Reward_pi[state] = P[state][action][0][2] * P[state][action][0][0]
	while np.max(np.abs(value_function_old - value_function)) > tol :
		value_function_old = value_function.copy()
		for state in range(nS) :
			action = policy[state] 
			s_next = P[state][action][0][1]
			# 执行动作后两种状态，要么转移成功要么转移失败
			sum_ = P[state][action][0][0] * value_function_old[s_next] + (1 - P[state][action][0][0]) * value_function_old[state]
			# sum_ = value_function_old[s_hat]
			value_function[state] = Reward_pi[state] + gamma * sum_
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""
	策略改进
	args:
		P, nS, nA, gamma:
			defined at beginning of file
		value_from_policy: np.ndarray
			The value calculated from the policy
		policy: np.array
			The previous policy.
	returns:
		new_policy: np.ndarray[nS]
			An array of integers. Each integer is the optimal action to take
			in that state according to the environment dynamics and the
			given value function.
	"""
	new_policy = np.zeros(nS, dtype='int')
	for state in range(nS) :
		val_max = -1
		for action in range(nA) : 
			s_hat = P[state][action][0][1]
			sum_ = P[state][action][0][0] * value_from_policy[s_hat] + (1 - P[state][action][0][0]) * value_from_policy[state]
			temp = P[state][action][0][2] * P[state][action][0][0] + gamma * sum_
			if temp >= val_max :
				val_max = temp
				new_policy[state] = action
	print("policy is ：", new_policy)
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	'''
	策略迭代算法
	args:
		P, nS, nA, gamma:
			defined at beginning of file
		tol: float
			tol parameter used in policy_evaluation()
	returns:
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
	'''

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	policy_old = np.zeros(nS, dtype=int)
	while True :
		value_function = policy_evaluation(P, nS, nA, policy_old)
		policy = policy_improvement(P, nS, nA, value_function, policy_old)
		if (policy == policy_old).all() :
			break
		else :
			policy_old = policy
	return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	'''
	值迭代算法
	args:
		P, nS, nA, gamma:
			defined at beginning of file
		tol: float
			Terminate value iteration when
				max |value_function(s) - prev_value_function(s)| < tol
	returns:
		value_function: np.ndarray[nS]
		policy: np.ndarray[nS]
	'''
	policy = np.zeros(nS, dtype=int)
	V_hat = np.zeros(nS)
	V = np.zeros(nS)
	V[:] = np.inf
	# 计算最优值函数
	while np.max(np.abs(V - V_hat)) > tol :
		V = V_hat.copy()
		for s in range(nS):
			for act in range(nA) : 
				s_hat = P[s][act][0][1]
				sum_ = P[s][act][0][0] * V[s_hat] + (1 - P[s][act][0][0]) * V[s]
				temp = P[s][act][0][2] * P[s][act][0][0] + gamma * sum_
				if temp > V_hat[s] :
					V_hat[s] = temp
		print("value:", V_hat)
	# 根据最优值函数得到最优策略
	for s in range(nS):
		val_max = 0
		for act in range(nA) : 
			s_hat = P[s][act][0][1]
			sum_ = P[s][act][0][0] * V[s_hat] + (1 - P[s][act][0][0]) * V[s]
			temp = P[s][act][0][2] * P[s][act][0][0] + gamma * sum_
			if temp >= val_max :
				val_max = temp
				policy[s] = act
		print("policy:", policy)
	return V, policy

def render_single(env, policy, max_steps=100):
	'''
	在环境中执行策略
	args:
		env: gym.core.Environment
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	'''
	episode_reward = 0
	ob = env.reset()
	for _ in range(max_steps):
		env.render()
		time.sleep(0.25)
		a = policy[ob]
		ob, rew, done, _ = env.step(a)
		episode_reward += rew
		if done:
			break
	env.render()
	if not done:
		print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
	else:
		print("Episode reward: %f" % episode_reward)



if __name__ == "__main__":
	'''
	策略迭代与值迭代二选一
	'''
	env = gym.make("Deterministic-4x4-FrozenLake-v0")
	#env = gym.make("Stochastic-4x4-FrozenLake-v0")
	#env = gym.make("Deterministic-8x8-FrozenLake-v0")

	# 策略迭代
	if args.policy:
		print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)
		V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
		render_single(env, p_pi, 100)
	else:
		# 值迭代
		print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)
		V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
		render_single(env, p_vi, 100)