import os
import gc
import os
import random
import struct
import sys
import sysconfig
import time

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from matplotlib.animation import PillowWriter
from torch import nn, optim
from AVATAR import AvatarUNRES
from utils import switch_order

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
# mplab = importlib.util.spec_from_file_location("mplab.name", "C:\Python311\Lib\site-packages\mayavi\mlab.py")

path = './proteinA/'
model_path = './model.pth'
matplotlib.use('TkAgg')

# sys.path.append('C:\Python311\Lib\site-packages\mayavi')
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

pos = switch_order(pos)
vel = switch_order(vel)
acc = switch_order(acc)
force = switch_order(force)

meta = torch.tensor(meta_info, device=device).reshape(1000, 9 * 32)
coords = torch.tensor(pos, device=device)
velocities = torch.tensor(vel, device=device)
accelerations = torch.tensor(acc, device=device)
forces = torch.tensor(force, device=device)

outmap_min, _ = torch.min(coords, dim=1, keepdim=True)
outmap_max, _ = torch.max(coords, dim=1, keepdim=True)
coords = (coords - outmap_min) / (outmap_max - outmap_min)
outmap_min, _ = torch.min(velocities, dim=1, keepdim=True)
outmap_max, _ = torch.max(velocities, dim=1, keepdim=True)
velocities = (velocities - outmap_min) / (outmap_max - outmap_min)
outmap_min, _ = torch.min(accelerations, dim=1, keepdim=True)
outmap_max, _ = torch.max(accelerations, dim=1, keepdim=True)
accelerations = (accelerations - outmap_min) / (outmap_max - outmap_min)
outmap_min, _ = torch.min(forces, dim=1, keepdim=True)
outmap_max, _ = torch.max(forces, dim=1, keepdim=True)
forces = (forces - outmap_min) / (outmap_max - outmap_min)

train_size = int(0.9 * meta.shape[0])

meta, meta_test = meta[:train_size], meta[train_size:]

grid_step = 0.01
grid_padding = 2
center_id = [1.]
center_id_length = len(center_id)
num_of_space_parameters = 9 + center_id_length

coords_spaceXtime = torch.zeros(
    (coords.shape[0], grid_padding + int(1. / grid_step), grid_padding + int(1. / grid_step),
     grid_padding + int(1. / grid_step)), device=device)
velocities_spaceXtime = torch.zeros(
    (velocities.shape[0], grid_padding + int(1. / grid_step), grid_padding + int(1. / grid_step),
     grid_padding + int(1. / grid_step)), device=device)
accelerations_spaceXtime = torch.zeros(
    (accelerations.shape[0], grid_padding + int(1. / grid_step), grid_padding + int(1. / grid_step),
     grid_padding + int(1. / grid_step)), device=device)
forces_spaceXtime = torch.zeros(
    (forces.shape[0], grid_padding + int(1. / grid_step), grid_padding + int(1. / grid_step),
     grid_padding + int(1. / grid_step)), device=device)

# print(sys.getsizeof(coords_spaceXtime),sys.getsizeof(velocities_spaceXtime),sys.getsizeof(accelerations_spaceXtime),sys.getsizeof(forces_spaceXtime))

for i in range(coords.shape[0] - 1):
    coords_slice = coords[i].reshape(-1, 3)
    velocities_slice = velocities[i].reshape(-1, 3)
    accelerations_slice = accelerations[i].reshape(-1, 3)
    forces_slice = forces[i].reshape(-1, 3)

    coords_idx = (coords_slice / grid_step).type(torch.long)
    velocities_idx = (velocities_slice / grid_step).type(torch.long)
    accelerations_idx = (accelerations_slice / grid_step).type(torch.long)
    forces_idx = (forces_slice / grid_step).type(torch.long)

    coords_spaceXtime[i, coords_idx[:, 0], coords_idx[:, 1], coords_idx[:, 2]] = 1.
    velocities_spaceXtime[i, velocities_idx[:, 0], velocities_idx[:, 1], velocities_idx[:, 2]] = 1.
    accelerations_spaceXtime[i, accelerations_idx[:, 0], accelerations_idx[:, 1], accelerations_idx[:, 2]] = 1.
    forces_spaceXtime[i, forces_idx[:, 0], forces_idx[:, 1], forces_idx[:, 2]] = 1.

# print(coords_spaceXtime.shape, velocities_spaceXtime.shape, accelerations_spaceXtime.shape, forces_spaceXtime.shape)
coords_spaceXtime,coords_spaceXtime_test = coords_spaceXtime[:train_size],coords_spaceXtime[train_size:]
velocities_spaceXtime,velocities_spaceXtime_test = velocities_spaceXtime[:train_size],velocities_spaceXtime[train_size:]
accelerations_spaceXtime,accelerations_spaceXtime_test = accelerations_spaceXtime[:train_size],accelerations_spaceXtime[train_size:]
forces_spaceXtime,forces_spaceXtime_test = forces_spaceXtime[:train_size],forces_spaceXtime[train_size:]

# coords, coords_test = coords[:train_size], coords[train_size:]
# velocities, velocities_test = velocities[:train_size], velocities[train_size:]
# accelerations, accelerations_test = accelerations[:train_size], accelerations[train_size:]
# forces, forces_test = forces[:train_size], forces[train_size:]

# Training loop
indices = torch.randperm(meta.shape[0] - 1)
train_sizev2 = int(0.95 * meta.shape[0])
val_size = meta.shape[0] - train_sizev2
train_indices = indices[:train_sizev2]
val_indices = indices[train_sizev2:]

num_epochs = 200
batch_size = 5
bloss = []
bbloss = []
model = AvatarUNRES(meta, coords_spaceXtime, velocities_spaceXtime, accelerations_spaceXtime, forces_spaceXtime).to(device)
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.HuberLoss(reduction='mean', delta=1.0)
lr = 1e-4
lr_mod = 1
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_idx = 0
for epoch in range(num_epochs):
    torch.set_grad_enabled(True)
    model.train()
    model.batch_size = batch_size
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr

    # RANDOM POINTS DYNAMIC LEARNING WITH STEP SIZE 1
    # t = random.sample(range(0, meta.shape[0] - 1), batch_size)
    t = torch.randperm(train_indices.numel())[:batch_size]
    t_1 = [s.item() + 1 for s in t]
    c_train, v_train, a_train, f_train = model(meta[t], coords_spaceXtime[t], velocities_spaceXtime[t],
                                               accelerations_spaceXtime[t], forces_spaceXtime[t])
    preds_train = torch.cat([c_train, v_train, a_train, f_train], dim=1)
    target_train = torch.cat([coords_spaceXtime[t_1], velocities_spaceXtime[t_1], accelerations_spaceXtime[t_1], forces_spaceXtime[t_1]], dim=1)
    # RANDOM POINTS DYNAMIC LEARNING WITH STEP SIZE 1

    # if (epoch) % 50 == 0:
    # SEQUENTIAL LEARNING WITH BATCH SIZE 1
    loss_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_coords_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_velocities_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_accelerations_seq = torch.tensor([0.], requires_grad=True, device=device)
    loss_forces_seq = torch.tensor([0.], requires_grad=True, device=device)

    # seq_len = random.randint(4, 50)
    seq_len = 5
    model.batch_size = 1
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr * lr_mod
    k = random.randint(0, coords_spaceXtime.shape[0] - seq_len - 1)
    seq_step_0 = range(k, k + seq_len)
    seq_step_1 = range(k + 1, k + seq_len + 1)
    c, v, a, f = coords_spaceXtime[seq_step_0], velocities_spaceXtime[seq_step_0], accelerations_spaceXtime[seq_step_0], forces_spaceXtime[seq_step_0]
    # for i in range(k, k + seq_len):
    # seq_step_0 = i
    # seq_step_1 = i + 1
    c_seq, v_seq, a_seq, f_seq = model(meta[seq_step_0], c, v, a, f, 1)
    preds_train_seq = torch.cat([c_seq, v_seq, a_seq, f_seq], dim=1)
    target_train_seq = torch.cat(
        [coords_spaceXtime[seq_step_1], velocities_spaceXtime[seq_step_1], accelerations_spaceXtime[seq_step_1],
         forces_spaceXtime[seq_step_1]], dim=1)
    loss_seq = loss_seq + criterion(preds_train_seq, target_train_seq)
    loss_coords_seq = loss_coords_seq + criterion(c_seq, coords_spaceXtime[seq_step_1])
    loss_velocities_seq = loss_velocities_seq + criterion(v_seq, velocities_spaceXtime[seq_step_1])
    loss_accelerations_seq = loss_accelerations_seq + criterion(a_seq, accelerations_spaceXtime[seq_step_1])
    loss_forces_seq = loss_forces_seq + criterion(f_seq, forces_spaceXtime[seq_step_1])
    # SEQUENTIAL LEARNING WITH BATCH SIZE 1

    loss_c = criterion(c_train, coords_spaceXtime[t_1]) + loss_coords_seq
    loss_v = criterion(v_train, velocities_spaceXtime[t_1]) + loss_velocities_seq
    loss_a = criterion(a_train, accelerations_spaceXtime[t_1]) + loss_accelerations_seq
    loss_f = criterion(f_train, forces_spaceXtime[t_1]) + loss_forces_seq
    separate_losess = [loss_c, loss_v, loss_a, loss_f]

    loss_idx = random.randint(0, 3)
    loss = loss_seq + criterion(preds_train, target_train) + sum(separate_losess)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch) % 10 == 0:
        with torch.set_grad_enabled(False):
            model.eval()
            tval = val_indices
            # tval = random.sample(range(0, meta.shape[0] - 1), batch_size)
            tval_1 = [s.item() + 1 for s in tval]
            model.batch_size = coords_spaceXtime[tval].shape[0]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            c_test, v_test, a_test, f_test = model(meta[tval], coords_spaceXtime[tval], velocities_spaceXtime[tval],
                                                   accelerations_spaceXtime[tval],
                                                   forces_spaceXtime[tval])
            preds_test = torch.cat([c_test, v_test, a_test, f_test], dim=1)
            target_test = torch.cat([coords_spaceXtime[tval_1], velocities_spaceXtime[tval_1], accelerations_spaceXtime[tval_1], forces_spaceXtime[tval_1]], dim=1)
            loss_val = criterion(preds_test, target_test)
            bloss.append(loss.item())
            bbloss.append(loss_val.item())
            if epoch > 20 and loss_val < max(bbloss) and loss < max(bloss):
                torch.save(model.state_dict(), model_path)
                # batch_size += 1
                # if batch_size > 100:
                #     batch_size -= 1
            print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {loss.item():.5f}, Validation Loss: {loss_val:.5f}')

c, v, a, f = torch.unsqueeze(coords_spaceXtime_test[0], dim=0), torch.unsqueeze(velocities_spaceXtime_test[0], dim=0), torch.unsqueeze(
    accelerations_spaceXtime_test[0], dim=0), torch.unsqueeze(forces_spaceXtime_test[0], dim=0)

fig = plt.figure()
# plt.style.use('dark_background')
plt.plot(bloss, c='blue')
plt.plot(bbloss, c='orange')
plt.grid()
plt.show()

pred_dynamics_pos = []
pred_dynamics_vel = []
pred_dynamics_acc = []
pred_dynamics_force = []

gtdlen = meta_test.shape[0] * 1
model.batch_size = 1
for param_group in optimizer.param_groups:
    param_group['lr'] = lr * lr_mod
model.eval()
start = time.time()
with torch.set_grad_enabled(False):
    for i in range(int(gtdlen)):
        c, v, a, f = model(meta_test[i].unsqueeze(0), c, v, a, f)
        print(c)
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
    ground_truth_dynamics_pos.append(coords_spaceXtime_test[i].cpu())
    ground_truth_dynamics_vel.append(velocities_spaceXtime_test[i].cpu())
    ground_truth_dynamics_acc.append(accelerations_spaceXtime_test[i].cpu())
    ground_truth_dynamics_force.append(forces_spaceXtime_test[i].cpu())

ground_truth_dynamics_pos = np.array(ground_truth_dynamics_pos)
ground_truth_dynamics_vel = np.array(ground_truth_dynamics_vel)
ground_truth_dynamics_acc = np.array(ground_truth_dynamics_acc)
ground_truth_dynamics_force = np.array(ground_truth_dynamics_force)

fig = plt.figure()
plt.style.use('dark_background')
prota = fig.add_subplot(111, projection='3d')
prota.set_xlabel("x")
prota.set_ylabel("y")
prota.set_zlabel("z")
marker_size = 1.5
linewidth = 1
ims = []

############# MLAB
# fig = mlab.figure()
# x = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# y = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# z = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# xp = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# yp = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# zp = np.random.rand(int(ground_truth_dynamics_pos.shape[1] / 3))
# plt = mlab.points3d(x, y, z, color=(1, 0, 0), scale_factor=0.07)

# @mlab.animate()
# def update_anim():
############# MLAB

for i in range(int(gtdlen)):
    k = 0
    x, y, z, xp, yp, zp = [], [], [], [], [], []
    for j in range(0, int(pred_dynamics_pos.shape[1] / 3)):
        # print(pred_dynamics_pos.shape)
        x.append(ground_truth_dynamics_pos[i, k])
        y.append(ground_truth_dynamics_pos[i, k + 1])
        z.append(ground_truth_dynamics_pos[i, k + 2])
        xp.append(pred_dynamics_pos[i, k])
        yp.append(pred_dynamics_pos[i, k + 1])
        zp.append(pred_dynamics_pos[i, k + 2])
        k += 3

    folding = prota.scatter(x, y, z, linewidth=linewidth, antialiased=False, s=marker_size, c="blue")
    folding_pred = prota.scatter(xp, yp, zp, linewidth=linewidth, antialiased=False, s=marker_size, c="red")

    ims.append([folding, folding_pred])

    # plt.mlab_source.set(x=xp, y=yp, z=zp)
    # yield

#
# update_anim()
# mlab.show()

ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat=True)

# ani.save("proteinA-folding_3d_v1_.gif", dpi=600, writer=PillowWriter(fps=20))
plt.show()

model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
