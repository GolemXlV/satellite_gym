from gym.envs.registration import register
from pathlib import Path
import pandas as pd
from random import randint

COLUMNS = ['id', 'sat_id', 'x_sim', 'y_sim', 'z_sim', 'Vx_sim', 'Vy_sim', 'Vz_sim']
SATELLITES_NUM = 300

df = pd.read_csv(Path('./envs/data/train.csv'), index_col='id', usecols=COLUMNS)
df = df[df['sat_id'] == randint(0, SATELLITES_NUM)]  # take random satellite
df.drop('sat_id', inplace=True, axis=1)
df.reset_index(drop=True, inplace=True)


register(
     id='SatelliteEnv-v1',
     entry_point='envs.satellite_env:SatelliteEnv',
     max_episode_steps=len(df.loc[:].values),
     kwargs={'df': df},
)
