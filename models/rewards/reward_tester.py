import os
import sys
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from cpv_reward import CPV
from baseline_reward import Baseline

sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

data = 'data/babyai'
train_master = 'data/babyai/train_master.json'
with open(train_master) as file:
    master = json.load(file)

print(os.listdir())

cpv_stuff = os.path.join('models', 'rewards/babyai_cpv_contrast_loss.pth')
cpv_reward = CPV(torch.load(cpv_stuff))

progress = []
rewards = []
fake_rewards = []

for i in range(100):
    print(i)
    ex = master[i]
    task_folder = ex['folder']
    task_file = ex['file']
    task_num = ex['ann']


    lang_root = os.path.join(data, task_folder, task_file + ".json")
    img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")

    with open(lang_root) as file:
        d = json.load(file)[task_num]
        img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
        imgs = img_file["arr_" + str(task_num)]
        imgs = np.split(imgs, len(imgs) // 7)
        img_file.close()

    mission = d['lang_raw']
    num_imgs = len(imgs)

    seen_imgs = [imgs[0]]
    del imgs[0]
    for im in range(len(imgs)):
        reward, _ = cpv_reward.calculate_reward(mission, seen_imgs, imgs[im])
        fake = np.trunc((np.random.rand(7, 7, 3) * 225))
        fake_reward, _ = cpv_reward.calculate_reward(mission, seen_imgs, fake)
        seen_imgs.append(imgs[im])

        progress.append(im/num_imgs)
        rewards.append(reward)
        fake_rewards.append(fake_reward)

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(progress, rewards, s=0.1, c='b')
print("a")
ax1.scatter(progress, fake_rewards, s=0.1, c='r')
print("b")
plt.xlabel("Progress Through Task")
plt.ylabel("Reward Value")
plt.ylim(-5, 5)

plt.savefig("reward_cpv_test_contrast_loss_percent.png")
