### Simple OpenAI Gym Environments for Satellite Prediction Tasks

#### How to use:
> git clone this rep

> cd satellite_gym && pip install -e .

#### Then create this environment in your python code:

> import gym, envs

> env = gym.make('SatelliteEnv-v1')

PS: I scaled all features and spaces with MinMax between 0 and 10. No thanks.