import gym
import numpy as np
import pandas as pd


LOW = -1E5
HIGH = 1E8
FEATURE_LOW = 0
FEATURE_HIGH = 10


class SatelliteEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(SatelliteEnv, self).__init__()
        self.df = self._transform(df)

        space_shape = (1, self.df.shape[-1])
        self.action_space = gym.spaces.Box(low=FEATURE_LOW, high=FEATURE_HIGH, shape=space_shape, dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=FEATURE_LOW, high=FEATURE_HIGH, shape=space_shape, dtype=np.float64)

        self.current_step = 0
        self.reward = 0

    def step(self, action: np.array):

        self.current_step += 1
        reward = self.reward
        done = False

        if self.current_step >= len(self.df.loc[:].values):
            self.current_step = 0
            reward = self.reward = 0
            done = True

        if not done:
            reward = self.reward = self._calc_reward(action, self.df.loc[self.current_step].values)

        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.reward = 0
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        frame = self.df.loc[self.current_step].values
        return frame

    def _calc_reward(self, satellite_predicted_values, satellite_true_values):
        # SMAPE
        return np.mean(np.abs((satellite_predicted_values - satellite_true_values)
                              / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values))))

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scale = (FEATURE_HIGH - FEATURE_LOW)

        columns = list(df.select_dtypes('number').columns)

        for column in columns:
            normalized_column = (df[column] - LOW) / (HIGH - LOW) * scale

            args = {column: normalized_column + FEATURE_LOW}

            df = df.assign(**args)

        return df

    def render(self, mode='human', close=False):
        print(f'Step: {self.current_step}')
        print(f'Reward: {self.reward}')
        print(f'Values: {self.df.loc[self.current_step].values}')
