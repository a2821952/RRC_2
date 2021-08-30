from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import os
from three_wolves.envs.utilities import model_utils
from three_wolves.envs.contact_cube_env import PhaseControlEnv
from trifinger_simulation.tasks import move_cube_on_trajectory as mct
import numpy as np
import argparse


def eval_model(model, model_name):
    score = []
    print('----------------------------------')
    print('Result of ', model_name)
    for i in range(5):
        obs = env.reset()
        done = False
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
        t_score = info['eval_score']
        score.append(t_score)
        print(f'step {i} : {t_score}')
    print('mean score: ', np.mean(score))
    print('----------------------------------')


if __name__ == "__main__":
    MODEL_NAME = 'tg_0'

    log_dir = f"three_wolves/deep_whole_body_controller/model_save/{MODEL_NAME}/"
    env = PhaseControlEnv(goal_trajectory=None,
                          visualization=True)
    model_utils.plot_results(log_dir)
    md = SAC.load(log_dir + "best_model.zip")
    eval_model(md, MODEL_NAME)
