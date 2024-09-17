import math
import random

import numpy as np
import indicators


NUMBER_OF_HIDDEN_NEURONS = 8


def sigmoid_activate(layer):
    return 1 / (1 + np.exp(-layer))


def softmax_activate(layer):
    m = np.exp(layer)
    return m / m.sum(len(layer.shape) - 1)


class NeuralNetwork:
    def __init__(self, observation_size=6, is_copy=False, including_position=False, normalization_parameters=None, validation_size=0, chances_of_mutation=1e-1):
        self._rewards = 0
        self._layers = []
        self._biases = []
        self.order_number = 0
        self.including_position = including_position
        self.extra_neuron = 0
        if including_position:
            self.extra_neuron = 1
        self.validation_size = validation_size
        self._validation_rewards = [0 for i in range(validation_size)]
        self._type = 'random'
        self.chances_of_mutation = chances_of_mutation
        self.neurons = NUMBER_OF_HIDDEN_NEURONS
        #self.observation_size = observation_size
        if normalization_parameters is None:
            normalization_parameters = {}
            for name in indicators.supported_indicators:
                normalization_parameters[name] = [1, 0]
        else:
            self.normalization_parameters = normalization_parameters
        nb_layers = 2

        if not is_copy:
            self.indicator_names = indicators.get_random_size_indicators()
            for i in range(nb_layers):
                entry_size = self.neurons if i != 0 else len(self.indicator_names)+self.extra_neuron
                self._layers.append(np.random.rand(self.neurons, entry_size) * 2 - 1)
                self._biases.append(np.random.rand(self.neurons, 1) * 2 - 1)

            self._outputs = np.random.rand(3, self.neurons) * 2 - 1

    def forward(self, input_dict):
        inputs = np.array([(input_dict[name] - self.normalization_parameters[name][1]) / (self.normalization_parameters[name][0] - self.normalization_parameters[name][1]) for name in self.indicator_names])
        if self.including_position:
            inputs = np.append(inputs, input_dict['position'])
        inputs = inputs.reshape((-1, 1))

        for layer, bias in zip(self._layers, self._biases):
            inputs = np.matmul(layer, inputs)
            inputs = inputs + bias
            inputs = sigmoid_activate(inputs)

        inputs = np.matmul(self._outputs, inputs)
        inputs = inputs.reshape(-1)

        return softmax_activate(inputs)

    def backward(self):
        pass

    def mutate(self):
        new_network = NeuralNetwork(normalization_parameters=self.normalization_parameters, validation_size=self.validation_size, is_copy=True)
        new_network.set_validation_rewards(self.get_validation_rewards().copy())
        new_network._type = 'mutate'

        new_network.indicator_names = self.indicator_names.copy()
        r = random.random()

        if r < 0.5:
            self.mutate_network(new_network)
        elif r < 0.666:
            new_indicator = indicators.get_one_random_indicator(self.indicator_names)
            random_index = random.randint(0, len(new_network.indicator_names) - 1)
            new_network.indicator_names[random_index] = new_indicator
            self.mutate_network(new_network)
        elif (r < 0.832 and len(self.indicator_names) > indicators.COMBO_LOWER_BOUND) or len(self.indicator_names) == indicators.COMBO_UPPER_BOUND:
            random_index = random.randint(0, len(self.indicator_names) - 1)
            new_network.indicator_names.remove(self.indicator_names[random_index])
            new_network._layers.append(np.delete(self._layers[0], random_index, axis=1))
            new_network._biases.append(self._biases[0])
            self.mutate_network(new_network, start_layer_index=1)
        else:
            new_indicator = indicators.get_one_random_indicator(self.indicator_names)
            new_network.indicator_names.insert(0, new_indicator)
            l = np.concatenate((np.random.rand(self.neurons, 1) * 2 - 1, self._layers[0]), axis=1)
            new_network._layers.append(l)
            new_network._biases.append(self._biases[0])
            self.mutate_network(new_network, start_layer_index=1)

        return new_network

    def mutate_network(self, new_network, start_layer_index=0):
        for i in range(start_layer_index, len(self._layers)):
            l = self._layers[i]
            random_mutation_probs = np.random.rand(l.shape[0], l.shape[1])

            random_mutation_probs = np.where(random_mutation_probs < self.chances_of_mutation,
                                             (np.random.rand() - 0.5) / 2, 0)
            new_l = l + random_mutation_probs
            new_network._layers.append(new_l)

        for i in range(start_layer_index, len(self._biases)):
            b = self._biases[i]
            random_mutation_probs = np.random.rand(b.shape[0], 1)
            random_mutation_probs = np.where(random_mutation_probs < self.chances_of_mutation,
                                             (np.random.rand() - 0.5) / 2, 0)
            new_l = b + random_mutation_probs
            new_network._biases.append(new_l)

        random_mutation_probs = np.random.rand(self._outputs.shape[0], self._outputs.shape[1])
        random_mutation_probs = np.where(random_mutation_probs < self.chances_of_mutation,
                                         (np.random.rand() - 0.5) / 2, 0)

        new_l = self._outputs + random_mutation_probs
        new_network._outputs = new_l

    def mutate_features(self):
        self.indicator_names = indicators.get_random_size_indicators()

    def crossover(self, other):
        for i in range(len(self._layers)):
            half = len(self._layers[i]) // 2
            # print(self._layers[i], self._layers[i][:half], other._layers[i][half:])
            self._layers[i] = np.concatenate((self._layers[i][:half], other._layers[i][half:]), axis=0)
            # print(self._layers[i])

        for i in range(len(self._biases)):
            half = len(self._biases[i]) // 2
            self._biases[i] = np.concatenate((self._biases[i][:half], other._biases[i][half:]), axis=0)

        half = len(self._outputs) // 2
        self._outputs = np.concatenate((self._outputs[:half], other._outputs[:half]), axis=0)

    def set_reward(self, r):
        self._rewards = r

    def get_reward(self):
        return self._rewards

    def set_validation_reward(self, index, r):
        self._validation_rewards[index] = r

    def get_validation_reward(self, index):
        return self._validation_rewards[index]

    def set_validation_rewards(self, r):
        self._validation_rewards = r

    def get_validation_rewards(self):
        return self._validation_rewards


def crossover3(agent1, agent2):
    new_network = NeuralNetwork(normalization_parameters=agent1.normalization_parameters, validation_size=agent1.validation_size, is_copy=True)
    new_network.set_validation_rewards(agent1.get_validation_rewards().copy())
    new_network._type = 'crossover'

    random_point = random.randint(0, NUMBER_OF_HIDDEN_NEURONS)
    random_point1 = random.randint(0, len(agent1.indicator_names))
    random_point2 = random.randint(0, len(agent2.indicator_names))
    if random_point1 + (len(agent2.indicator_names)-random_point2) < indicators.COMBO_LOWER_BOUND:
        random_point1 = indicators.COMBO_LOWER_BOUND - (len(agent2.indicator_names)-random_point2)
    new_indicator_names = agent1.indicator_names[:random_point1]
    new_l = agent1._layers[0][:, :random_point1]
    new_b = np.concatenate((agent1._biases[0][:random_point], agent2._biases[0][random_point:]), axis=0)
    for i in range(len(agent2.indicator_names) - random_point2):
        if len(new_indicator_names) >= indicators.COMBO_UPPER_BOUND:
            break
        if agent2.indicator_names[random_point2 + i] not in new_indicator_names:
            new_indicator_names.append(agent2.indicator_names[random_point2 + i])
            new_l = np.concatenate((new_l, agent2._layers[0][:, [random_point2 + i]]), axis=1)
    new_network._layers.append(new_l)
    new_network._biases.append(new_b)
    new_network.indicator_names = new_indicator_names

    for i in range(1, len(agent1._layers)):
        new_l = np.concatenate((agent1._layers[i][:random_point], agent1._layers[i][random_point:]), axis=0)
        new_network._layers.append(new_l)

    for i in range(1, len(agent1._biases)):
        new_b = np.concatenate((agent1._biases[i][:random_point], agent1._biases[i][random_point:]), axis=0)
        new_network._biases.append(new_b)

    random_point = random.randint(0, 3)
    new_l = np.concatenate((agent1._outputs[:random_point], agent1._outputs[random_point:]), axis=0)
    new_network._outputs = new_l

    return new_network
