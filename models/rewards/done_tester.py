import os
import sys
import torch
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from cpv_reward import CPV
from baseline_reward import Baseline

import revtok
from vocab import Vocab


sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'models'))

data = 'data/babyai_complex_subset'
train_master = 'data/babyai_complex_subset/valid_master.json'
with open(train_master) as file:
    master = json.load(file)

print(os.listdir())

cpv_stuff = os.path.join('exp', 'model:cpv_babyai', 'pseudo_best.pth')
cpv_reward = CPV(torch.load(cpv_stuff))

# data = 'data/babyai_complex_subset'
# train_master = 'data/babyai_complex_subset/valid_master.json'
#
# with open(train_master) as file:
#     master = json.load(file)

progress = []
progress_new = []
rewards = []
# fake_rewards = []
rewards_new = []

j = random.randint(0, len(master))
ex = master[j]
task_folder = ex['folder']
task_file = ex['file']
task_num = ex['ann']

lang_root = os.path.join(data, task_folder, task_file + ".json")
img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")

with open(lang_root) as file:
    d = json.load(file)[task_num]
    num_instr = d['num_instr']
    norm_instr = d['lang_raw']

norm_instr = cpv_reward.calculate_reward(norm_instr, None, None)

rewards = num_instr
rewards_new = norm_instr
progress = list(range(len(rewards)))
progress_new = list(range(len(rewards_new)))



# for i in range(100):
#     print(i)
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         imgs = img_file["arr_" + str(task_num)]
#         imgs = np.split(imgs, len(imgs) // 7)
#         img_file.close()
#
#
#     imgs.append(np.zeros((7,7,3), dtype=np.intc))
#     mission = d['num_instr']
#     num_imgs = len(imgs)

# for i in range(10):
#     print(i)
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         imgs = img_file["arr_" + str(task_num)]
#         imgs = np.split(imgs, len(imgs) // 7)
#         img_file.close()
#
#     imgs.append(np.zeros((7,7,3), dtype=np.intc))
#     mission = d['num_instr']
#     num_imgs = len(imgs)
#
#
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         fake_imgs = img_file["arr_" + str(task_num)]
#         fake_imgs = np.split(fake_imgs, len(fake_imgs) // 7)
#         img_file.close()
#
#     seen_imgs = [[imgs[0]] for i in range(num_imgs)]
#     # fake_imgs = [fake]
#     del imgs[0]
#     for im in range(num_imgs - 1):
#         fake_idx = 0
#         for ni in range(num_imgs - 1):
#             if (ni > im):
#                 reward, done, norm = cpv_reward.calculate_reward(mission, seen_imgs[ni], imgs[im])
#                 seen_imgs[ni].append(imgs[im])
#                 progress.append(im/num_imgs)
#                 rewards.append(reward)
#             else:
#                 # fake_1 = np.trunc((np.random.rand(7, 7) * 11)).astype(np.int16)
#                 # fake_2 = np.trunc((np.random.rand(7, 7) * 6)).astype(np.int16)
#                 # fake_3 = np.trunc((np.random.rand(7, 7) * 3)).astype(np.int16)
#                 # fake = np.stack([fake_1, fake_2, fake_3], axis=-1)
#                 fake = fake_imgs[fake_idx]
#                 fake_reward, fake_done, fake_norm = cpv_reward.calculate_reward(mission, seen_imgs[ni], fake)
#                 seen_imgs[ni].append(fake)
#                 progress_new.append(im/num_imgs)
#                 rewards_new.append(fake_reward)
#         if fake_idx < len(fake_imgs) - 1:
#             fake_idx += 1
#
#

# for i in range(100):
#     print(i)
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         imgs = img_file["arr_" + str(task_num)]
#         imgs = np.split(imgs, len(imgs) // 7)
#         img_file.close()
#
#
#     imgs.append(np.zeros((7,7,3), dtype=np.intc))
#     mission = d['num_instr']
#     num_imgs = len(imgs)
#
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         fake_imgs = img_file["arr_" + str(task_num)]
#         fake_imgs = np.split(fake_imgs, len(fake_imgs) // 7)
#         img_file.close()
#
#
#     seen_imgs = [imgs[0]]
#     fake_seen_imgs = [fake_imgs[0]]
#
#     del imgs[0]
#     del fake_imgs[0]
#     for im in range(len(imgs)):
#         reward, done, norm = cpv_reward.calculate_reward(mission, seen_imgs, imgs[im])
#         # fake = np.trunc((np.random.rand(7, 7, 3) * 225))
#         seen_imgs.append(imgs[im])
#         progress.append(im/num_imgs)
#         rewards.append(done)
#
#     for im in range(len(fake_imgs)):
#         fake_reward, fake_done, fake_norm = cpv_reward.calculate_reward(mission, fake_seen_imgs, fake_imgs[im])
#         fake_seen_imgs.append(fake_imgs[im])
#         progress_new.append(im/(len(fake_imgs) + 1))
#         rewards_new.append(fake_done)



# data = 'data/babyai_complex_subset'
# train_master = 'data/babyai_complex_subset/valid_master.json'
#
# with open(train_master) as file:
#     master = json.load(file)
#
#
# for i in range(100):
#     print(i)
#     j = random.randint(0, len(master))
#     ex = master[j]
#     task_folder = ex['folder']
#     task_file = ex['file']
#     task_num = ex['ann']
#
#
#     lang_root = os.path.join(data, task_folder, task_file + ".json")
#     img_root = os.path.join(data, task_folder, "imgs" + task_file[4:] + ".npz")
#
#     with open(lang_root) as file:
#         d = json.load(file)[task_num]
#         img_file = np.load(img_root) # sizes of images are 7 x 7 x 3
#         imgs = img_file["arr_" + str(task_num)]
#         imgs = np.split(imgs, len(imgs) // 7)
#         img_file.close()
#
#     imgs.append(np.zeros((7,7,3), dtype=np.intc))
#     mission = d['num_instr']
#     num_imgs = len(imgs)
#
#
#     seen_imgs = [imgs[0]]
#     del imgs[0]
#     for im in range(len(imgs)):
#         reward, done, norm = cpv_reward.calculate_reward(mission, seen_imgs, imgs[im])
#         # fake = np.trunc((np.random.rand(7, 7, 3) * 225))
#         # fake_reward = cpv_reward.calculate_reward(mission, seen_imgs, fake)
#         seen_imgs.append(imgs[im])
#         progress_new.append(im/num_imgs)
#         rewards_new.append(done)
#         # fake_rewards.append(fake_reward)
#


fig = plt.figure()
ax1 = fig.add_subplot(111)


ax1.plot(progress, rewards, c='b')
ax1.plot(progress_new, rewards_new, c='r')

# ax1.scatter(progress_new, rewards_new, s=0.1, c='r')
# print("b")
# ax1.scatter(progress, rewards, s=0.1, c='b')
# print("a")
plt.xlabel("Progress Through Task")
plt.ylabel("Reward")

plt.savefig("cpv_babyai_1.png")
