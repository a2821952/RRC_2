import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


tp = np.load('center_complement.npy')
tc = np.load('small_complement.npy')
tcr = np.load('random_complement.npy')
data_list = [tp, tc, tcr]

reward_means = [np.mean(d, 0)[0] for d in data_list]
reward_std = [np.std(d, 0)[0] for d in data_list]
score_means = [np.mean(d, 0)[1] for d in data_list]
score_std = [np.std(d, 0)[1] for d in data_list]

data_dict = {
    'models': ['TP', 'TP + CP', 'TP + CP + DR'],
    'reward_mean': reward_means,
    'reward_std': reward_std,
    'score_mean': score_means,
    'score_std': score_std,
}

df = pd.DataFrame(data_dict)

# ax = df.plot(x="models", y=['reward', 'score'], kind="bar")
ax_0 = plt.subplot(211)
ax_0.bar(df['models'], df['reward_mean'], color='orange', alpha=0.5,
         yerr=df['reward_std'], align='center', ecolor='black', capsize=10)
plt.ylabel('RL Reward', fontsize=15)
plt.grid(axis='y')
plt.xticks(df['models'], fontsize=15)

ax_1 = plt.subplot(212)
ax_1.bar(df['models'], df['score_mean'], color='blue', alpha=0.5,
         yerr=df['score_std'], align='center', ecolor='black', capsize=10)
plt.ylabel('Competition Reward', fontsize=15)
plt.grid(axis='y')
plt.xticks([])
plt.tight_layout()
plt.savefig('cube_tg_models.pdf')
plt.show()

# real score
# rl: -8804.604467780237 -5620.34492504529 -38880.770977753506 -40132.61292910406
# random: -12904.46905335215 -19836.69031076311 -25235.13351354389
# old: -8771.74330647553 -7698.399187188018 -42316.004429756395

