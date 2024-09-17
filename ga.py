import numpy as np

import indicators
from fin_env import FinancialEnvironment
import fin_env
from neural_network import NeuralNetwork
import neural_network
import random
from matplotlib import pyplot as plt
import pickle
from joblib import Parallel, delayed


NUMBER_OF_POPULATION = 100


def load_agent(size=1):
    agents = []
    for i in range(size):
        with open("result/agent" + str(i) + ".obj", 'rb') as file_save:
            agent_params = pickle.load(file_save)

            agent = NeuralNetwork(normalization_parameters=agent_params["normalization"])
            agent._layers = agent_params["layers"]
            agent._biases = agent_params["biases"]
            agent._outputs = agent_params["outputs"]
            agent.indicator_names = agent_params["indicators"]
            agents.append(agent)
    return agents


def save_agent(agents):
    for i in range(len(agents)):
        agent = agents[i]
        with open("result/agent" + str(i) + ".obj", 'wb') as file_save:
            # We create a dict of the snake parameters
            to_save = {"layers": agent._layers,
                       "biases": agent._biases,
                       "outputs": agent._outputs,
                       "indicators": agent.indicator_names,
                       "normalization": agent.normalization_parameters}
            # And save it using picke
            pickle.dump(to_save, file_save, protocol=pickle.HIGHEST_PROTOCOL)


def simulation(agent, data):
    env = FinancialEnvironment(data, simulate_trade=False)
    #total_reward = 0  # Used to track the performance of the agent.

    observation = env.reset()
    done = False

    # action_history = []
    total_reward = 0
    while not done:
        action = agent.forward(observation).argmax() - 1
        observation, reward, done, info = env.step(action)
        # action_history.append(action)
        total_reward = info

    # print(action_history)
    return [agent.order_number, total_reward]


def sort_reward(reward_array):
    sorted_reward_array = sorted(reward_array, key=lambda x:x[0])
    return [item[1] for item in sorted_reward_array]


def train(data, generation, validation_set=None):
    camp = []
    max_reward = np.sum(np.abs(data['roc1']))
    print(max_reward)

    # normalize
    normalization_parameters = {}
    for indicator_name in indicators.supported_indicators:
        indicator_max = data[indicator_name].max()
        indicator_min = data[indicator_name].min()
        # data[indicator_name] = data[indicator_name] / indicator_max
        normalization_parameters[indicator_name] = [indicator_max, indicator_min]

    for i in range(NUMBER_OF_POPULATION):
        agent = NeuralNetwork(normalization_parameters=normalization_parameters, validation_size=(len(validation_set)))
        camp.append(agent)

    percentage_saved = 0.2
    qte_to_keep = int(percentage_saved * len(camp))

    number_of_iterations = generation
    reward_history, total_history, mutation_history, crossover_history = [], [], [], []
    if validation_set is not None:
        validate_reward_history = [[] for _ in range(len(validation_set))]
    for num in range(number_of_iterations):
        '''for i in range(len(camp)):
            camp[i].order_number = i
            if len(validate_reward_history) > 0 and len(validate_reward_history[0]) > 0:
                for j in range(len(validate_reward_history)):
                    camp[i].set_validation_reward(j, validate_reward_history[j][len(validate_reward_history[j])-1])'''

        total_reward_array = sort_reward(Parallel(n_jobs=16)(delayed(simulation)(agent, data) for agent in camp[qte_to_keep:]))

        if validation_set is not None:
            validation_reward_all = []
            for validation_data in validation_set:
                validation_reward_all.append(sort_reward(Parallel(n_jobs=16)(delayed(simulation)(agent, validation_data) for agent in camp[qte_to_keep:])))
        for i in range(len(camp)-qte_to_keep):
            agent = camp[i + qte_to_keep]
            if validation_set is None:
                agent.set_reward(total_reward_array[i])
            else:
                validate_result = True
                for j in range(len(validation_reward_all)):
                    if validation_reward_all[j][i] <= agent.get_validation_reward(j):
                        validate_result = False
                if validate_result:
                    agent.set_reward(total_reward_array[i])
                    for j in range(len(validation_reward_all)):
                        agent.set_validation_reward(j, validation_reward_all[j][i])
                else:
                    agent.set_reward(0)
        '''for agent in camp:
            total_reward = 0  # Used to track the performance of the agent.

            observation = env.reset()
            done = False

            while not done:
                action = agent.forward(observation).argmax() - 1
                observation, reward, done, info = env.step(action)
                total_reward += reward

            agent.set_reward(total_reward)'''

        camp.sort(key=lambda agent1: agent1.get_reward()+sum(agent1.get_validation_rewards()), reverse=True)
        #print([camp[i].get_reward() for i in range(len(camp))])
        reward_history.append(camp[0].get_reward())
        total_reward = camp[0].get_reward()
        if validation_set is not None:
            for i in range(len(validation_set)):
                validate_reward_history[i].append(camp[0].get_validation_reward(i))
                total_reward += camp[0].get_validation_reward(i)
        total_history.append(total_reward)
        if camp[0]._type == 'mutate':
            mutation_history.append(total_reward)
            crossover_history.append(None)
        elif camp[0]._type == 'crossover':
            mutation_history.append(None)
            crossover_history.append(total_reward)
        else:
            mutation_history.append(None)
            crossover_history.append(None)
        # print(camp[0]._layers, camp[0]._biases)

        print(f'generation {num}, {camp[0]._type}, reward={camp[0].get_reward()}, v_reward={camp[0].get_validation_rewards()}, indicators={camp[0].indicator_names}')


        qte_new = len(camp) - qte_to_keep
        new_camp = camp[:qte_to_keep]  # The agents we'll keep.
        for agent in new_camp:
            agent._type = 'elite'

        new_species = []
        for i in range(qte_new // 2):
            father_agent = new_camp[i % qte_to_keep]
            new_agent = father_agent.mutate()
            new_species.append(new_agent)
        for i in range(qte_new // 2):
            agent1 = camp[random.randint(0, qte_to_keep - 1)]
            agent2 = camp[random.randint(0, qte_to_keep - 1)]
            new_agent = neural_network.crossover3(agent1, agent2)
            new_species.append(new_agent)
        camp = new_camp + new_species

    env = FinancialEnvironment(data, simulate_trade=False)
    observation = env.reset()
    done = False
    action_history = []
    while not done:
        action = agent.forward(observation).argmax() - 1
        observation, reward, done, info = env.step(action)
        action_history.append(action)
    print(action_history)

    x = np.arange(len(reward_history))
    plt.plot(x, total_history, color='r', label='profit(%)')
    plt.scatter(x, mutation_history, color='b', label='mutation')
    plt.scatter(x, crossover_history, color='g', label='crossover')
    plt.legend()
    plt.savefig('result/training.png')

    return [camp[0], reward_history, validate_reward_history]


def train_all(train_frames, validation_frames, test_frames, generation=100):
    # result = Parallel(n_jobs=4)(delayed(train)(data, generation) for data in dataframes[:-1])
    result = []
    for i in range(len(train_frames)):
        result.append(train(train_frames[i], generation, validation_set=validation_frames[i]))
    best_agents = [x[0] for x in result]
    best_reward_histories = [x[1] for x in result]
    best_validate_reward_histories = [x[2] for x in result]

    #fig, axs = plt.subplots(2, 2)
    #for i in range(len(best_reward_histories)):
    #    axs[i // 2, i % 2].plot(best_reward_histories[i])
    x = np.arange(len(best_reward_histories[0]))
    plt.clf()
    plt.plot(x, best_reward_histories[0], color='r', label='training')
    for i in range(len(best_validate_reward_histories[0])):
        plt.plot(x, best_validate_reward_histories[0][i], label='validation')
    plt.legend()
    plt.savefig('result/train.png')

    save_agent(best_agents)

    for i in range(len(test_frames)):
        test_all(best_agents, test_frames[i], i)


def test_all(agents, test_frame, index):
    # test best agents
    test_env = FinancialEnvironment(test_frame, simulate_trade=False)
    test_length = len(test_frame)

    hold_reward = np.sum(test_frame['roc1'])
    print(hold_reward)

    x = np.arange(test_length)
    plt.clf()

    test_env.reset()
    action_index, done, capital_history = 0, False, [0]
    while not done:
        observation, reward, done, info = test_env.step(1)
        capital_history.append(info)

    plt.plot(x, capital_history, color='r', label='benchmark')

    mixed_actions = np.zeros(test_length)
    acc_reward_history = np.zeros(test_length)
    for i in range(len(agents)):
        #print(agents[i].get_reward(), agents[i].get_validation_rewards())
        observation = test_env.reset()
        first_agent, done, action_history, capital_history = agents[i], False, [], [0]
        precision = 0
        action_index = 0
        while not done:
            action = first_agent.forward(observation).argmax() - 1
            mixed_actions[action_index] += action
            observation, reward, done, info = test_env.step(action)
            action_history.append(action)
            acc_reward = capital_history[len(capital_history) - 1] + reward
            capital_history.append(info)
            acc_reward_history[action_index + 1] += acc_reward
            if reward > 0:
                precision += 1
            action_index += 1

        #print(action_history)
        #print(precision / test_length)
        #print(first_agent.indicator_names)
        print(acc_reward_history[len(acc_reward_history)-1])
        #plt.plot(x, capital_history, label='period ' + str(i + 1))
        plt.plot(x, capital_history, label='GA-ANN model')

    if len(agents)>1:
        #plt.plot(x, acc_reward_history / len(agents), label='period avg')

        test_env.reset()
        action_index, done, capital_history = 0, False, [0]
        while not done:
            action = mixed_actions[action_index]
            if action > 1:
                action = 1
            elif action < -1:
                action = -1
            observation, reward, done, info = test_env.step(action)
            capital_history.append(info)
            action_index += 1

        plt.plot(x, capital_history, label='period mix')

    plt.legend()
    plt.savefig(f'result/reward{index}.png')
