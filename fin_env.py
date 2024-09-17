import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import indicators


INITIAL_CAPITAL = 10000


class Observation:
    def __init__(self):
        pass


class FinancialEnvironment:
    def __init__(self, training_df, simulate_trade=False, transaction_cost=0, leverage=1):
        self.capital = 0
        self._state = 0
        self.stock_df = training_df
        self.simulate_trade = simulate_trade
        self.transaction_cost = transaction_cost
        self.leverage = leverage
        self.current_position = 0
        self.current_position_capital = 0
        self.observation_attributes = indicators.supported_indicators
        self._action_spec = np.array([1, 0, -1])
        self._observation_spec = np.zeros((len(self.observation_attributes),))
        self.reset_vars()

    def reset_vars(self):
        if self.simulate_trade:
            self.capital = INITIAL_CAPITAL
        else:
            self.capital = 0
        self.current_position = 0
        self.current_position_capital = 0
        self._day_counter = 0
        self._state = 0
        self.max_loss = 10
        self._total_reward = 0
        self._waiting = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def get_current_observation(self):
        observations = self.stock_df[self.observation_attributes].iloc[self._day_counter].to_dict()
        observations['position'] = self.current_position
        return observations

    def get_current_reward(self):
        return self.stock_df.iloc[self._day_counter]['roc1'] * self.current_position

    def reset(self):
        self.reset_vars()
        return self.get_current_observation()

    def update_df(self, attribute_name, attribute_value):
        self.stock_df.at[self.stock_df.index.values[self._day_counter], attribute_name] = attribute_value

    def get_position_capital(self):
        if self.capital > INITIAL_CAPITAL:
            position_capital = INITIAL_CAPITAL * self.leverage
        else:
            position_capital = self.capital * self.leverage
        return position_capital

    def step(self, action):
        self._day_counter += 1

        if self.simulate_trade:
            reward = 0
            if self.current_position * action == -1:
                reward -= self.current_position_capital*self.transaction_cost
                self.current_position_capital = self.get_position_capital()
                reward += self.current_position_capital*self.stock_df.iloc[self._day_counter]['roc1']/100 * self.current_position
                reward -= self.current_position_capital*self.transaction_cost
                self.capital += reward
                self.current_position = action
            elif self.current_position == 0 and action != 0:
                self.current_position_capital = self.get_position_capital()
                reward -= self.current_position_capital * self.transaction_cost
                self.capital += reward
                self.current_position = action
            elif self.current_position != 0 and action == 0:
                reward -= self.current_position_capital * self.transaction_cost
                self.current_position_capital = self.get_position_capital()
                reward += self.current_position_capital * self.stock_df.iloc[self._day_counter][
                    'roc1'] / 100 * self.current_position
                self.capital += reward
                self.current_position = action
            elif self.current_position != 0 and self.current_position == action:
                reward += self.current_position_capital * self.stock_df.iloc[self._day_counter]['roc1']/100 * self.current_position
                self.capital += reward

            if self.capital < 0:
                self.capital = 0
                self.current_position_capital = 0
            observations = self.get_current_observation()
        else:
            self.current_position = action
            observations = self.get_current_observation()
            reward = self.get_current_reward()
            self.capital += reward

        # if self._total_reward > self.max_loss:
        # return ts.termination(observations, reward=reward)

        if self._day_counter + 1 == len(self.stock_df):
            return observations, reward, True, self.capital
        else:
            return observations, reward, False, self.capital
