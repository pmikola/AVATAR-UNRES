import math
import os
import random
import struct
import sys
import sysconfig
import time
from statistics import mean

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt, cm, animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import axes3d
from torch import nn, optim
from torch.nn.functional import normalize

from AVATAR import AvatarUNRES
import gc

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)

path = './proteinA/'
model_path = './model.pth'
matplotlib.use('TkAgg')

print(sysconfig.get_paths()["purelib"])
sys.path.append('C:/Python311/Lib/site-packages')
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA is available! GPU will be used.")
else:
    device = torch.device('cpu')
    print("CUDA is not available. CPU will be used.")
# device = torch.device('cpu')

######### proteinA.x file loading
prota_MD_NVE_noxdr_MD000 = open(path + "prota_MD_NVE-noxdr_MD000.x", "r")
data = prota_MD_NVE_noxdr_MD000.read()
splited_data = data.split()

meta_info = []
pos = []
vel = []
acc = []
force = []
temp = []
switch = 0
i = 0
for _ in range(len(splited_data)):
    if i >= len(splited_data):
        break
    if switch == 0:
        temp.append(splited_data[i])
        i += 1
    if splited_data[i] == "#coordinates":
        meta_info.append(temp)
        del temp
        temp = []
        switch = 1
        i += 1
    if switch == 1:
        temp.append(splited_data[i])
        i += 1
    if splited_data[i] == "#velocities":
        pos.append(temp)
        del temp
        temp = []
        switch = 2
        i += 1
    if switch == 2:
        temp.append(splited_data[i])
        i += 1
    if splited_data[i] == "#accelerations":
        vel.append(temp)
        del temp
        temp = []
        switch = 3
        i += 1
    if switch == 3:
        temp.append(splited_data[i])
        i += 1
    if splited_data[i] == "#potential":
        i += 1
    if splited_data[i] == "forces":
        acc.append(temp)
        del temp
        temp = []
        switch = 4
        i += 1
        j = 0
    if switch == 4:
        temp.append(splited_data[i])
        i += 1
        j += 1
        if j % len(acc[0]) == 0:
            force.append(temp)
            del temp
            temp = []
            switch = 0

meta_info = [[float(s) for s in x] for x in meta_info]
pos = [[float(s) for s in x] for x in pos]
vel = [[float(s) for s in x] for x in vel]
acc = [[float(s) for s in x] for x in acc]
force = [[float(s) for s in x] for x in force]

def binary(num):
    # IEEE 754
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))


for i in range(len(meta_info[:])):
    m2 = []
    for j in range(len(meta_info[0])):
        m0 = [*binary(meta_info[i][j])]
        m1 = [float(s) for s in m0]
        m2.append(m1)
    meta_info[i] = m2

meta = torch.tensor(meta_info, device=device).reshape(1000, 9 * 32)
coords = torch.tensor(pos, device=device)
velocities = torch.tensor(vel, device=device)
accelerations = torch.tensor(acc, device=device)
forces = torch.tensor(force, device=device)

train_size = int(0.9 * meta.shape[0])
meta, meta_test = meta[:train_size], meta[train_size:]
coords, coords_test = coords[:train_size], coords[train_size:]
velocities, velocities_test = velocities[:train_size], velocities[train_size:]
accelerations, accelerations_test = accelerations[:train_size], accelerations[train_size:]
forces, forces_test = forces[:train_size], forces[train_size:]

# meta = normalize(meta, p=1.)
# coords = normalize(coords, p=1.)
# velocities = normalize(velocities, p=1.)
# accelerations = normalize(accelerations, p=1.)
# forces = normalize(forces, p=1.)

######### proteinA.x file loading
######### proteinA.txt inertia file loading
# proteinA_inertia = open(path + "proteinA-inertia.txt", "r")
# data_inertia = proteinA_inertia.read()
# splited_data_inertia = data_inertia.split()
# TODO: do this after successful model
# print(splited_data_inertia)
######### proteinA.txt inertia file loading
t_step = 4.89e-15

model = AvatarUNRES(meta, coords, velocities, accelerations, forces).to(device)
criterion = nn.MSELoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
indices = torch.randperm(meta.shape[0] - 1)
train_sizev2 = int(0.95 * meta.shape[0])
val_size = meta.shape[0] - train_sizev2
train_indices = indices[:train_sizev2]
val_indices = indices[train_sizev2:]

num_epochs = 1
batch_size = 25
bloss = []
bbloss = []

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    # model.eval()

loss_idx = 0
for epoch in range(num_epochs):
    torch.set_grad_enabled(True)
    model.train()
    model.batch_size = batch_size
    # RANDOM POINTS DYNAMIC LEARNING
    # t = random.sample(range(0, meta.shape[0] - 1), batch_size)
    t = torch.randperm(train_indices.numel())[:batch_size]
    t_1 = [s.item() + 1 for s in t]
    c_train, v_train, a_train, f_train = model(meta[t], coords[t], velocities[t], accelerations[t], forces[t])
    preds_train = torch.cat([c_train, v_train, a_train, f_train], dim=1)
    target_train = torch.cat([coords[t_1], velocities[t_1], accelerations[t_1], forces[t_1]], dim=1)
    # RANDOM POINTS DYNAMIC LEARNING

    # if (epoch) % 50 == 0:
    # Sequential learning
    loss_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_coords_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_velocities_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_accelerations_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_forces_seq = torch.tensor([0.], requires_grad=True, device=device)

    seq_len = random.randint(4, 50)
    k = random.randint(0, coords.shape[0] - seq_len-1)
    c, v, a, f = torch.unsqueeze(coords[k], dim=0), torch.unsqueeze(velocities[k], dim=0), torch.unsqueeze(
        accelerations[k], dim=0), torch.unsqueeze(forces[k], dim=0)
    model.batch_size = 1
    for i in range(k, k + seq_len):
        c_seq, v_seq, a_seq, f_seq = model(meta[i].unsqueeze(0), c, v, a, f)
        preds_train_seq = torch.cat([c_seq, v_seq, a_seq, f_seq], dim=1)
        target_train_seq = torch.cat(
            [torch.unsqueeze(coords[i + 1], dim=0), torch.unsqueeze(velocities[i + 1], dim=0),
             torch.unsqueeze(accelerations[i + 1], dim=0), torch.unsqueeze(forces[i + 1], dim=0)], dim=1)
        loss_seq = loss_seq + criterion(preds_train_seq, target_train_seq)
        loss_coords_seq = loss_coords_seq + criterion(c_seq, torch.unsqueeze(coords[i + 1], dim=0))
        loss_velocities_seq = loss_velocities_seq + criterion(v_seq, torch.unsqueeze(velocities[i + 1], dim=0))
        loss_accelerations_seq = loss_accelerations_seq + criterion(a_seq, torch.unsqueeze(accelerations[i + 1], dim=0))
        loss_forces_seq = loss_forces_seq + criterion(f_seq, torch.unsqueeze(forces[i + 1], dim=0))
        # Sequential learning

    loss_c = criterion(c_train, coords[t_1]) + loss_coords_seq
    loss_v = criterion(v_train, velocities[t_1]) + loss_velocities_seq
    loss_a = criterion(a_train, accelerations[t_1]) + loss_accelerations_seq
    loss_f = criterion(f_train, forces[t_1]) + loss_forces_seq
    separate_losess = [loss_c, loss_v, loss_a, loss_f]

    loss_idx = random.randint(0, 3)
    loss = separate_losess[loss_idx] + loss_seq + criterion(preds_train, target_train)
    bloss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch) % 50 == 0:
        with torch.set_grad_enabled(False):
            model.batch_size = batch_size
            tval = val_indices
            # tval = random.sample(range(0, meta.shape[0] - 1), batch_size)
            tval_1 = [s.item() + 1 for s in tval]

            c_test, v_test, a_test, f_test = model(meta[tval], coords[tval], velocities[tval], accelerations[tval],
                                                   forces[tval])
            preds_test = torch.cat([c_test, v_test, a_test, f_test], dim=1)
            target_test = torch.cat([coords[tval_1], velocities[tval_1], accelerations[tval_1], forces[tval_1]], dim=1)
            loss_val = criterion(preds_test, target_test)
            bbloss.append(loss_val)
            if epoch > 100 and loss_val < max(bbloss) and loss < max(bloss):
                torch.save(model.state_dict(), model_path)
                batch_size += 1
                if batch_size > 100:
                    batch_size -= 1
            print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {loss.item():.5f}, Validation Loss: {loss_val:.5f}')

c, v, a, f = torch.unsqueeze(coords_test[0], dim=0), torch.unsqueeze(velocities_test[0], dim=0), torch.unsqueeze(
    accelerations_test[0],
    dim=0), torch.unsqueeze(
    forces_test[0], dim=0)
pred_dynamics_pos = []
pred_dynamics_vel = []
pred_dynamics_acc = []
pred_dynamics_force = []

gtdlen = meta_test.shape[0]*1
model.batch_size = 1

model.eval()
start = time.time()
with torch.set_grad_enabled(False):
    for i in range(int(gtdlen)):
        c, v, a, f = model(meta_test[i].unsqueeze(0), c, v, a, f)
        pred_dynamics_pos.append(c.detach().cpu().numpy())
        pred_dynamics_vel.append(v.detach().cpu().numpy())
        pred_dynamics_acc.append(a.detach().cpu().numpy())
        pred_dynamics_force.append(f.detach().cpu().numpy())
end = time.time()
print((end - start) / int(gtdlen))  # speed
pred_dynamics_pos = np.array(pred_dynamics_pos).squeeze(1)

pred_dynamics_vel = np.array(pred_dynamics_vel).squeeze(1)
pred_dynamics_acc = np.array(pred_dynamics_acc).squeeze(1)
pred_dynamics_force = np.array(pred_dynamics_force).squeeze(1)

ground_truth_dynamics_pos = []
ground_truth_dynamics_vel = []
ground_truth_dynamics_acc = []
ground_truth_dynamics_force = []

for i in range(int(gtdlen)):
    ground_truth_dynamics_pos.append(coords_test[i].cpu())
    ground_truth_dynamics_vel.append(velocities_test[i].cpu())
    ground_truth_dynamics_acc.append(accelerations_test[i].cpu())
    ground_truth_dynamics_force.append(forces_test[i].cpu())

ground_truth_dynamics_pos = np.array(ground_truth_dynamics_pos)
ground_truth_dynamics_vel = np.array(ground_truth_dynamics_vel)
ground_truth_dynamics_acc = np.array(ground_truth_dynamics_acc)
ground_truth_dynamics_force = np.array(ground_truth_dynamics_force)

x = range(np.array(ground_truth_dynamics_pos).shape[1])
y = range(int(gtdlen))
# fig = plt.figure(figsize=(8, 6))
# plt.style.use('dark_background')
# hp = fig.add_subplot(111, projection='3d')
# hv = fig.add_subplot(222, projection='3d')
# ha = fig.add_subplot(223, projection='3d')
# hf = fig.add_subplot(224, projection='3d')
X, Y = np.meshgrid(x, y)
marker = '.'
marker_size = 0.1
linewidth = 0.1

# hp.scatter(X, Y, pred_dynamics_pos, linewidth=linewidth, antialiased=False, s=marker_size, c="orange")
# hv.scatter(X, Y, pred_dynamics_vel, linewidth=linewidth, antialiased=False, s=marker_size, c="orange")
# ha.scatter(X, Y, pred_dynamics_acc, linewidth=linewidth, antialiased=False, s=marker_size, c="orange")
# hf.scatter(X, Y, pred_dynamics_force, linewidth=linewidth, antialiased=False, s=marker_size, c="orange")

# hp.scatter(X, Y, ground_truth_dynamics_pos, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")
# hv.scatter(X, Y, ground_truth_dynamics_vel, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")
# ha.scatter(X, Y, ground_truth_dynamics_acc, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")
# hf.scatter(X, Y, ground_truth_dynamics_force, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")

# hp.set_xlabel("Coordinates xyz")
# hp.set_ylabel("Time step")
# hp.set_zlabel("Coordinates values")


# plt.tight_layout(pad=-5.0, w_pad=-5.0, h_pad=-5.0)
# hp.grid(False)
# hp.axis('off')
# hv.grid(False)
# hv.axis('off')
# ha.grid(False)
# ha.axis('off')
# hf.grid(False)
# hf.axis('off')

# plt.show()

# time.sleep(1000)
# animate protein coordinates

fig = plt.figure(figsize=(8, 6))
# plt.style.use('dark_background')
prota = fig.add_subplot(111, projection='3d')
prota.set_xlabel("x")
prota.set_ylabel("y")
prota.set_zlabel("z")
marker_size = 1
linewidth = 1
ims = []

for i in range(int(gtdlen)):
    xx = 0
    yy = 14
    zz = 29
    x, y, z, xp, yp, zp = [], [], [], [], [], []
    for j in range(0, int(pred_dynamics_pos.shape[1] / 3)):
        # print(pred_dynamics_pos.shape)
        x.append(ground_truth_dynamics_pos[i, xx])
        y.append(ground_truth_dynamics_pos[i, yy])
        z.append(ground_truth_dynamics_pos[i, zz])
        xp.append(pred_dynamics_pos[i, xx])
        yp.append(pred_dynamics_pos[i, yy])
        zp.append(pred_dynamics_pos[i, zz])

        xx += 1
        yy += 1
        zz += 1
        if zz % 15 == 0:
            xx += 15
            yy += 15
            zz += 15

    folding = prota.scatter(x, y, z, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")
    folding_pred = prota.scatter(xp, yp, zp, linewidth=linewidth, antialiased=False, s=marker_size, c="orange")

    ims.append([folding,folding_pred])

# ani = animation.ArtistAnimation(fig, ims, interval=30, blit=True, repeat=True)
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat=True)

# ani.save("proteinA-folding.gif", dpi=300, writer=PillowWriter(fps=30))
plt.show()

model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
