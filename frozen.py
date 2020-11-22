# This code is my submission for CS 7641 Assignment 4: Markov Decision Processes. All analysis and experimenting done by me. 
# Zach Sirera - Fall 2020

import numpy as np
import gym
from gym import wrappers
import time
import sys
import matplotlib.pyplot as plt



def make_patch_spines_invisible(ax):
	''' this is used to get the three y axes on the policy iteration and value iteration plots ''' 
	ax.set_frame_on(True)
	ax.patch.set_visible(False)
	for sp in ax.spines.values():
		sp.set_visible(False)


def make_env(environment_name: str):
	''' make the environment that the different algorithms will operate on '''
	env = gym.make(environment_name)
	env = env.unwrapped
	desc = env.unwrapped.desc

	return env, desc


def frozen_lake_pi(env, desc):
	''' carry out policy iteration on the frozen lake environment '''

	policy_results = []

	gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	# gammas = [1.0]
	for gamma in gammas:

		start_time = time.time()
		best_policy = policy_iteration(env, gamma = gamma)
		score = evaluate_policy(env, best_policy, gamma = gamma)
		duration = time.time() - start_time

		# print(best_policy)

		policy_results.append({'gamma': gamma, 'score': np.mean(score), 'time': duration}) #'iters': k, 
	
	fig, ax_1 = plt.subplots()
	fig.subplots_adjust(right=0.75)

	ax_2 = ax_1.twinx()
	ax_3 = ax_1.twinx()

	ax_3.spines["right"].set_position(("axes", 1.2))
	make_patch_spines_invisible(ax_3)
	ax_3.spines["right"].set_visible(True)


	p1, = ax_1.plot([x['gamma'] for x in policy_results], [y['time'] for y in policy_results], label="Duration", c="r")
	p2, = ax_2.plot([x['gamma'] for x in policy_results], [y['score'] for y in policy_results], label="Avg Score", c="b")
	p3, = ax_3.plot([x['gamma'] for x in policy_results], [y['iters'] for y in policy_results], label="Iterations", c="g")

	lines = [p1, p2, p3]
	
	ax_1.set_xlabel("Gamma")
	ax_1.set_ylabel("Duration (s)", c="r")
	ax_2.set_ylabel("Avg Score", c="b")
	ax_3.set_ylabel("No. of Iterations", c="g")

	ax_1.legend(lines, [l.get_label() for l in lines])

	plt.title("Frozen Lake: Policy Iteration")
	plt.legend(loc="upper right")
	plt.tight_layout()
	plt.savefig('figures/pi_lake.png')
	plt.clf()



def frozen_lake_vi(env, desc):
	''' carry out value iteration on the frozen lake environment ''' 
	# moves = {'left': 0; 'down': 1; 'right': 2; 'up': 3}

	value_results = []

	gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
	for gamma in gammas:

		start_time = time.time()

		best_value, k = value_iteration(env, gamma = gamma)
		policy = extract_policy(env,best_value, gamma = gamma)
		policy_score = evaluate_policy(env, policy, gamma=gamma, n=1000)

		duration = time.time() - start_time

		value_results.append({'gamma': gamma, 'iters': k, 'best_val': best_value, 'score': np.mean(policy_score), 'time': duration})


	# plot duration, score, and number of iterations as a function of gamma
	fig, ax_1 = plt.subplots()
	fig.subplots_adjust(right=0.75)

	ax_2 = ax_1.twinx()
	ax_3 = ax_1.twinx()

	ax_3.spines["right"].set_position(("axes", 1.2))
	make_patch_spines_invisible(ax_3)
	ax_3.spines["right"].set_visible(True)


	p1, = ax_1.plot([x['gamma'] for x in value_results], [y['time'] for y in value_results], label="Duration", c="r")
	p2, = ax_2.plot([x['gamma'] for x in value_results], [y['score'] for y in value_results], label="Avg Score", c="b")
	p3, = ax_3.plot([x['gamma'] for x in value_results], [y['iters'] for y in value_results], label="Iterations", c="g")

	lines = [p1, p2, p3]
	
	ax_1.set_xlabel("Gamma")
	ax_1.set_ylabel("Duration (s)", c="r")
	ax_2.set_ylabel("Avg Score", c="b")
	ax_3.set_ylabel("No. of Iterations", c="g")

	ax_1.legend(lines, [l.get_label() for l in lines])

	plt.title("Frozen Lake: Value Iteration")
	plt.legend(loc="upper right")
	plt.tight_layout()
	plt.savefig('figures/vi_lake.png')
	plt.clf()


	# plot the state values at each gamma 
	for each in value_results:
		plt.plot(each['best_val'], label=each['gamma'], lw=0, marker='o')

	plt.title("Frozen Lake: Value Iteration \n Gamma")
	plt.xlabel("State")
	plt.ylabel("State Value")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.savefig("figures/vi_lake_values.png")
	plt.clf()


def frozen_lake_vi_optimal():

	best_policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

	colors = {'S': 0, 'O': 1, 'H': 2, 'G': 3}
	lake = ['S', 'O', 'O', 'O', 'O', 'H', 'O', 'H', 'O', 'O', 'O', 'H', 'H', 'O', 'O', 'G']
	directions = {3: '⬆', 2: '➡', 1: '⬇', 0: '⬅'}


	grid_moves = [[0 for i in range(4)] for i in range(4)]
	grid_values = [[0 for i in range(4)] for i in range(4)]
	grid_lake = [[0 for i in range(4)] for i in range(4)]

	fig, ax = plt.subplots()

	for i in range(len(best_policy)):
		x = i % 4
		y = i // 4

		grid_lake[y][x] = colors[lake[i]]

		text = ax.text(x, y + 0.15, lake[i], ha="center", va="center", color="k", fontsize=20)
		text = ax.text(x, y - 0.15, directions[best_policy[i]], ha="center", va="center", color="w", fontsize=20)


	plt.imshow(grid_lake)
	plt.title("Frozen Lake: Value Iteration \n Optimal Policy")
	plt.yticks([],[])
	plt.xticks([],[])
	plt.savefig("figures/vi_frozen_optimal_policy.png")
	plt.cla()
	# plt.show()


def frozen_lake_pi_optimal():

	best_policy = [0, 3, 3, 3, 0, 0, 0, 0, 3, 1, 0, 0, 0, 2, 1, 0]

	colors = {'S': 0, 'O': 1, 'H': 2, 'G': 3}
	lake = ['S', 'O', 'O', 'O', 'O', 'H', 'O', 'H', 'O', 'O', 'O', 'H', 'H', 'O', 'O', 'G']
	directions = {3: '⬆', 2: '➡', 1: '⬇', 0: '⬅'}

	grid_moves = [[0 for i in range(4)] for i in range(4)]
	grid_values = [[0 for i in range(4)] for i in range(4)]
	grid_lake = [[0 for i in range(4)] for i in range(4)]

	fig, ax = plt.subplots()

	for i in range(len(best_policy)):
		x = i % 4
		y = i // 4

		grid_lake[y][x] = colors[lake[i]]

		text = ax.text(x, y + 0.15, lake[i], ha="center", va="center", color="k", fontsize=20)
		text = ax.text(x, y - 0.15, directions[best_policy[i]], ha="center", va="center", color="w", fontsize=20)


	plt.imshow(grid_lake)
	plt.title("Frozen Lake: Policy Iteration \n Gamma = 1    20,000 Iterations")
	plt.yticks([],[])
	plt.xticks([],[])
	plt.savefig("figures/pi_frozen_optimal_policy.png")
	plt.cla()
	# plt.show()




	
def frozen_lake_q_gamma(env, desc):
	''' carry out Q-Learning on the frozen lake environment '''

	q_results = []

	start_time = time.time()


	gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for gamma in gammas:


		q_array = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal = [0 for i in range(env.observation_space.n)]
		epsilon = 0.85
		alpha = 0.4
		episodes = 1000

		env, desc = make_env('FrozenLake-v0')

		for episode in range(episodes):

			state = env.reset()
			done = False
			current_reward = 0
			max_steps = 1000

			for i in range(max_steps):
				if done:
					break        
				current = state
				if np.random.rand() < (epsilon):
					action = np.argmax(q_array[current, :])
				else:
					action = env.action_space.sample()
				
				state, reward, done, info = env.step(action)
				current_reward += reward
				q_array[current, action] += alpha * (reward + gamma * np.max(q_array[state, :]) - q_array[current, action])

			epsilon = (1 - 2.71 ** (-episode / 1000))
			rewards.append(current_reward)
			iters.append(i)


		for k in range(env.observation_space.n):
			optimal[k] = np.argmax(q_array[k, :])

		env.close()
		duration = time.time() - start_time

		def chunk_list(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		size = int(episodes / 50)
		all_chunks = list(chunk_list(rewards, size))

		q_results.append({
			'gamma': gamma,
			'rewards': rewards, 
			'iterations': iters, 
			'q_array': q_array, 
			'time': duration, 
			'size': size, 
			'chunks': all_chunks, 
			'averages': [sum(chunk) / len(chunk) for chunk in all_chunks]
			})
	


	for index, each in enumerate(q_results):
		plt.subplot(1, len(q_results), index + 1)
		var = plt.imshow(each['q_array'])
		if index == 0:
			plt.yticks([i for i in range(16)], [str(i + 1) for i in range(16)])
		else:
			plt.yticks([],[])


		plt.title(str(each['gamma']))
		plt.xticks([0, 1, 2, 3], ['left', 'down', 'right', 'up'], rotation=90)

	plt.suptitle("Frozen Lake: Q Learning \n Discount Factor")
	plt.tight_layout()
	plt.savefig('figures/ql_frozen_discount_factor.png')
	plt.clf()



def frozen_lake_q_epsilon(env, desc):
	''' carry out Q-Learning on the frozen lake environment '''

	q_results = []

	start_time = time.time()

	epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for epsilon in epsilons:

		q_array = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal = [0 for i in range(env.observation_space.n)]
		gamma = 0.9
		alpha = 0.4
		episodes = 1000

		env, desc = make_env('FrozenLake-v0')

		for episode in range(episodes):

			state = env.reset()
			done = False
			current_reward = 0
			max_steps = 1000

			for i in range(max_steps):
				if done:
					break        
				current = state
				if np.random.rand() < (epsilon):
					action = np.argmax(q_array[current, :])
				else:
					action = env.action_space.sample()
				
				state, reward, done, info = env.step(action)
				current_reward += reward
				q_array[current, action] += alpha * (reward + gamma * np.max(q_array[state, :]) - q_array[current, action])

			epsilon = (1 - 2.71 ** (-episode / 1000))
			rewards.append(current_reward)
			iters.append(i)


		for k in range(env.observation_space.n):
			optimal[k] = np.argmax(q_array[k, :])

		env.close()
		duration = time.time() - start_time

		def chunk_list(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		size = int(episodes / 50)
		all_chunks = list(chunk_list(rewards, size))

		q_results.append({
			'epsilon': epsilon,
			'rewards': rewards, 
			'iterations': iters, 
			'q_array': q_array, 
			'time': duration, 
			'size': size, 
			'chunks': all_chunks, 
			'averages': [sum(chunk) / len(chunk) for chunk in all_chunks]
			})
	


	for index, each in enumerate(q_results):
		plt.subplot(1, len(q_results), index + 1)
		var = plt.imshow(each['q_array'])
		if index == 0:
			plt.yticks([i for i in range(16)], [str(i + 1) for i in range(16)])
		else:
			plt.yticks([],[])


		plt.title(str(epsilons[index]))
		plt.xticks([0, 1, 2, 3], ['left', 'down', 'right', 'up'], rotation=90)

	plt.suptitle("Frozen Lake: Q Learning \n Exploration vs Exploitation: Initial Epsilon")
	plt.tight_layout()
	plt.savefig('figures/ql_frozen_epsilon.png')
	plt.clf()


def frozen_lake_q_alpha(env, desc):
	''' carry out Q-Learning on the frozen lake environment '''

	q_results = []

	start_time = time.time()
	# epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	# for epsilon in epsilons:

	alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	for alpha in alphas:


		q_array = np.zeros((env.observation_space.n, env.action_space.n))
		rewards = []
		iters = []
		optimal = [0 for i in range(env.observation_space.n)]
		epsilon = 0.85
		gamma = 0.9
		episodes = 1000

		env, desc = make_env('FrozenLake-v0')

		for episode in range(episodes):

			state = env.reset()
			done = False
			current_reward = 0
			max_steps = 1000

			for i in range(max_steps):
				if done:
					break        
				current = state
				if np.random.rand() < (epsilon):
					action = np.argmax(q_array[current, :])
				else:
					action = env.action_space.sample()
				
				state, reward, done, info = env.step(action)
				current_reward += reward
				q_array[current, action] += alpha * (reward + gamma * np.max(q_array[state, :]) - q_array[current, action])

			epsilon = (1 - 2.71 ** (-episode / 1000))
			rewards.append(current_reward)
			iters.append(i)


		for k in range(env.observation_space.n):
			optimal[k] = np.argmax(q_array[k, :])

		env.close()
		duration = time.time() - start_time

		def chunk_list(l, n):
			for i in range(0, len(l), n):
				yield l[i:i + n]

		size = int(episodes / 50)
		all_chunks = list(chunk_list(rewards, size))

		q_results.append({
			'alpha': alpha,
			'rewards': rewards, 
			'iterations': iters, 
			'q_array': q_array, 
			'time': duration, 
			'size': size, 
			'chunks': all_chunks, 
			'averages': [sum(chunk) / len(chunk) for chunk in all_chunks]
			})
	

	for index, each in enumerate(q_results):
		plt.subplot(1, len(q_results), index + 1)
		var = plt.imshow(each['q_array'])
		if index == 0:
			plt.yticks([i for i in range(16)], [str(i + 1) for i in range(16)])
		else:
			plt.yticks([],[])


		plt.title(str(each['alpha']))
		plt.xticks([0, 1, 2, 3], ['left', 'down', 'right', 'up'], rotation=90)

	plt.suptitle("Frozen Lake: Q Learning \n Learning Rates")
	plt.tight_layout()
	plt.savefig('figures/ql_frozen_learning_rates.png')
	plt.clf()


def frozen_lake_multi(env, desc):
	''' carry out Q-Learning on the frozen lake environment '''

	q_results = []

	gammas = [0.1 * i for i in range(1, 11)]
	alphas = [0.5, 1]
	epsilons = [0.5, 1]
	for gamma in gammas:
		for index, epsilon in enumerate(epsilons):
			for alpha in alphas:

				q_array = np.zeros((env.observation_space.n, env.action_space.n))
				rewards = []
				iters = []
				# epsilon = 1.0
				optimal = [0 for i in range(env.observation_space.n)]
				episodes = 1000

				env, desc = make_env('FrozenLake-v0')

				start_time = time.time()
				for episode in range(episodes):

					state = env.reset()
					done = False
					current_reward = 0
					max_steps = 1000

					for i in range(max_steps):
						if done:
							break        
						current = state
						if np.random.rand() < (epsilon):
							action = np.argmax(q_array[current, :])
						else:
							action = env.action_space.sample()
						
						state, reward, done, info = env.step(action)
						current_reward += reward
						q_array[current, action] += alpha * (reward + gamma * np.max(q_array[state, :]) - q_array[current, action])

					epsilon = (1 - 2.71 ** (-episode / 1000))
					rewards.append(current_reward)
					iters.append(i)


				for k in range(env.observation_space.n):
					optimal[k] = np.argmax(q_array[k, :])

				env.close()
				duration = time.time() - start_time
		

				q_results.append({
					'gamma': gamma,
					'epsilon': epsilons[index],
					'alpha': alpha,
					'time': duration
					})

	# print(q_results)

	x_1 = [x['gamma'] for x in q_results if x['alpha'] == 0.5 and x['epsilon'] == 0.5]
	y_1 = [y['time'] for y in q_results if y['alpha'] == 0.5 and y['epsilon'] == 0.5]

	x_2 = [x['gamma'] for x in q_results if x['alpha'] == 0.5 and x['epsilon'] == 1]
	y_2 = [y['time'] for y in q_results if y['alpha'] == 0.5 and y['epsilon'] == 1]

	x_3 = [x['gamma'] for x in q_results if x['alpha'] == 1 and x['epsilon'] == 0.5]
	y_3 = [y['time'] for y in q_results if y['alpha'] == 1 and y['epsilon'] == 0.5]

	x_4 = [x['gamma'] for x in q_results if x['alpha'] == 1 and x['epsilon'] == 1]
	y_4 = [y['time'] for y in q_results if y['alpha'] == 1 and y['epsilon'] == 1]


	plt.plot(x_1, y_1, label="alpha 0.5, epsilon 0.5")
	plt.plot(x_2, y_2, label="alpha 0.5, epsilon 1")
	plt.plot(x_3, y_3, label="aplha 1, epsilon 0.5")
	plt.plot(x_4, y_4, label="aplha 1, epsilon 1")
	plt.title("Frozen Lake: Q-Learning")
	plt.ylabel("Duration (s)")
	plt.xlabel("Gamma")
	plt.legend(loc="upper left")
	plt.tight_layout()
	plt.savefig("figures/ql_frozen_multi.png")


	

	


def frozen_lake_q_optimal(env, desc): 
	''' get the optimal policy using optimzed q learning '''


	q_array = np.zeros((env.observation_space.n, env.action_space.n))
	rewards = []
	iters = []
	alpha = 0.4
	epsilon = 0.85
	gamma = 0.9
	episodes = 1000

	for episode in range(episodes):

		state = env.reset()
		done = False
		current_reward = 0
		max_steps = 1000

		for i in range(max_steps):
			if done:
				break        
			current = state
			if np.random.rand() < (epsilon):
				action = np.argmax(q_array[current, :])
			else:
				action = env.action_space.sample()
			
			state, reward, done, info = env.step(action)
			current_reward += reward
			q_array[current, action] += alpha * (reward + gamma * np.max(q_array[state, :]) - q_array[current, action])

		epsilon = (1 - 2.71 ** (-episode / 1000))
		rewards.append(current_reward)
		iters.append(i)


	# plt.imshow(q_array)
	# plt.title("Optimal Q-Learning Average Value")
	# plt.ylabel("State")
	# plt.yticks([i for i in range(16)], [str(i + 1) for i in range(16)])
	# plt.xlabel("Move")
	# plt.xticks([0, 1, 2, 3], ['left', 'down', 'right', 'up'], rotation=90)
	# plt.savefig("figures/ql_frozen_optimal")
	# plt.clf()

	extract_q_policy(q_array)




def extract_q_policy(q_array):
	''' plot a grid/map for a given policy over the frozen lake environment '''

	moves = {'0': 'left', '1': 'down', '2': 'right', '3': 'up'}
	# colors = {'S': 'green', 'O': 'white', 'H': 'red', 'G': 'gold'}
	colors = {'S': 0, 'O': 1, 'H': 2, 'G': 3}
	lake = ['S', 'O', 'O', 'O', 
			'O', 'H', 'O', 'H', 
			'O', 'O', 'O', 'H', 
			'H', 'O', 'O', 'G']
	directions = {3: '⬆', 2: '➡', 1: '⬇', 0: '⬅'}

	policy = []

	for each in q_array:
		l = each.tolist()
		# policy.append(moves[str(l.index(max(l)))])
		policy.append(l.index(max(l)))


	grid_moves = [[0 for i in range(4)] for i in range(4)]
	grid_values = [[0 for i in range(4)] for i in range(4)]
	grid_lake = [[0 for i in range(4)] for i in range(4)]

	fig, ax = plt.subplots()

	for i in range(len(policy)):
		x = i % 4
		y = i // 4

		grid_lake[y][x] = colors[lake[i]]

		text = ax.text(x, y + 0.15, lake[i], ha="center", va="center", color="k", fontsize=20)
		text = ax.text(x, y - 0.15, directions[policy[i]], ha="center", va="center", color="w", fontsize=20)





	plt.imshow(grid_lake)
	plt.title("Frozen Lake: Q-Learning \n Optimal Policy")
	plt.yticks([],[])
	plt.xticks([],[])
	plt.savefig("figures/ql_frozen_optimal_policy.png")
	plt.cla()



def plot_empty_map():
	''' make an empty map for reference without policy movements in report  '''

	moves = {'0': 'left', '1': 'down', '2': 'right', '3': 'up'}
	# colors = {'S': 'green', 'O': 'white', 'H': 'red', 'G': 'gold'}
	colors = {'S': 0, 'O': 1, 'H': 2, 'G': 3}
	lake = ['S', 'O', 'O', 'O', 
			'O', 'H', 'O', 'H', 
			'O', 'O', 'O', 'H', 
			'H', 'O', 'O', 'G']
	directions = {3: '⬆', 2: '➡', 1: '⬇', 0: '⬅'}



	grid_moves = [[0 for i in range(4)] for i in range(4)]
	grid_values = [[0 for i in range(4)] for i in range(4)]
	grid_lake = [[0 for i in range(4)] for i in range(4)]

	fig, ax = plt.subplots()

	for i in range(len(lake)):
		x = i % 4
		y = i // 4

		grid_lake[y][x] = colors[lake[i]]

		text = ax.text(x, y, lake[i], ha="center", va="center", color="k", fontsize=20)



	plt.imshow(grid_lake)
	plt.title("Frozen Lake")
	plt.yticks([],[])
	plt.xticks([],[])
	plt.savefig("figures/frozen_map.png")
	plt.cla()





def run_episode(env, policy, gamma, render = True):
	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done , _ = env.step(int(policy[obs]))
		total_reward += (gamma ** step_idx * reward)
		step_idx += 1
		if done:
			break
	return total_reward



def evaluate_policy(env, policy, gamma , n = 100):
	scores = [run_episode(env, policy, gamma, False) for _ in range(n)]
	return np.mean(scores)



def extract_policy(env,v, gamma):
	policy = np.zeros(env.nS)
	for s in range(env.nS):
		q_sa = np.zeros(env.nA)
		for a in range(env.nA):
			q_sa[a] = sum([p * (r + gamma * v[s_]) for p, s_, r, _ in  env.P[s][a]])
		policy[s] = np.argmax(q_sa)
	return policy



def compute_policy_v(env, policy, gamma):
	v = np.zeros(env.nS)
	eps = 1e-5
	while True:
		prev_v = np.copy(v)
		for s in range(env.nS):
			policy_a = policy[s]
			v[s] = sum([p * (r + gamma * prev_v[s_]) for p, s_, r, is_done in env.P[s][policy_a]])
		if (np.sum((np.fabs(prev_v - v))) <= eps):
			break
	return v



def policy_iteration(env, gamma):
	policy = np.random.choice(env.nA, size=(env.nS))  
	max_iters = 20000
	desc = env.unwrapped.desc
	for i in range(max_iters):
		old_policy_v = compute_policy_v(env, policy, gamma)
		new_policy = extract_policy(env,old_policy_v, gamma)

		if (np.all(policy == new_policy)):
			k=i+1
			break
		policy = new_policy
	return policy #,k



def value_iteration(env, gamma):
	v = np.zeros(env.nS)  # initialize value-function
	max_iters = 100000
	eps = 1e-20
	desc = env.unwrapped.desc
	for i in range(max_iters):
		prev_v = np.copy(v)
		for s in range(env.nS):
			q_sa = [sum([p*(r + gamma*prev_v[s_]) for p, s_, r, _ in env.P[s][a]]) for a in range(env.nA)] 
			v[s] = max(q_sa)
		#if i % 50 == 0:
		#	plot = plot_policy_map('Frozen Lake Policy Map Iteration '+ str(i) + ' (Value Iteration) ' + 'Gamma: '+ str(gamma),v.reshape(4,4),desc,colors_lake(),directions_lake())
		if (np.sum(np.fabs(prev_v - v)) <= eps):
			k=i+1
			break
	return v,k



if __name__ == '__main__':
	env, desc = make_env('FrozenLake-v0')

	frozen_lake_pi(env, desc)
	frozen_lake_pi_optimal()
	frozen_lake_vi(env, desc)
	frozen_lake_vi_optimal()
	frozen_lake_q_gamma(env, desc)
	frozen_lake_q_alpha(env, desc)
	frozen_lake_q_epsilon(env, desc)
	frozen_lake_q_optimal(env, desc)
	frozen_lake_multi(env, desc)
	plot_empty_map()
	



