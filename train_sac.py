from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy
from stable_baselines3.common.monitor import Monitor
import os
from three_wolves.envs.utilities import model_utils
from three_wolves.envs.contact_cube_env import ContactControlEnv
from trifinger_simulation.tasks import move_cube_on_trajectory as mct
import numpy as np
import argparse


def get_arguments():
    parser = argparse.ArgumentParser("Tri-finger")
    # Environment
    parser.add_argument("--model-name", '-m', type=str, help="name of model(save path)")
    parser.add_argument("--run-mode", '-r', type=str, choices=['train', 'test'], help="train or test model")
    parser.add_argument("--device", '-d', type=int, default=0, help="cuda device")
    return parser.parse_args()



if __name__ == "__main__":
    args = get_arguments()
    log_dir = f"three_wolves/deep_whole_body_controller/model_save/{args.model_name}/"
    env_class = ContactControlEnv
    if args.run_mode == 'train':
        env = env_class(goal_trajectory=None,
                        visualization=False,
                        randomization='random' in args.model_name,
                        evaluation=False)
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, log_dir)
        callback = model_utils.SaveOnBestTrainingRewardCallback(check_freq=int(1e2), log_dir=log_dir)
        model = SAC(MlpPolicy, env, verbose=1, device=f'cuda:{args.device}')
        model.learn(total_timesteps=int(1e5), callback=callback, log_interval=int(1e2))
    else:
        env = env_class(goal_trajectory=None,
                        visualization=True,
                        randomization=True,
                        evaluation=True)
        model_utils.plot_results(log_dir)
        model = SAC.load(log_dir + "best_model.zip")
        rewards = []
        # rl_reward, score
        print('----------------------------------')
        print('Result of ', args.model_name)
        for i in range(5):
            obs = env.reset()
            done = False
            R = 0
            while not done:
                action, _states = model.predict(obs)
                obs, reward, done, info = env.step(action)
                R += reward
            t_score = info['eval_score']
            rewards.append([R, t_score])
            print(f'step {i} [reward : {R}, score : {t_score}]')
        print('mean reward: ', np.mean(rewards, 0)[0])
        print('mean score: ', np.mean(rewards, 0)[1])
        print('----------------------------------')

        # model_name = 'random_complement'
        # np.save(f'three_wolves/deep_whole_body_controller/evaluation/{model_name}.npy', rewards)

        # small:
        # mean reward:  24.360377381432738
        # mean score:  -10828.843981087892

        # contact
        # mean reward:  20.77457510631497
        # mean score:  -13725.570857651455

        # small random
        # mean reward:  17.49108352484235
        # mean score:  -14293.716150398957
