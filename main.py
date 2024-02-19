import math
import random
import struct
import sys
import sysconfig
from statistics import mean

import matplotlib
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import normalize

from AVATAR import AvatarUNRES
import gc

random.seed(2024)
np.random.seed(2024)
torch.manual_seed(2024)
path = './proteinA/'
matplotlib.use('TkAgg')
torch.manual_seed(2024)
print(sysconfig.get_paths()["purelib"])  # where python look for packages
sys.path.append('C:/Python311/Lib/site-packages')
if torch.cuda.is_available():
    device = torch.device('cuda')  # GPU available
    print("CUDA is available! GPU will be used.")
else:
    device = torch.device('cpu')  # No GPU available, fallback to CPU
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
criterion = nn.MSELoss(reduction='mean')
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Training loop
indices = torch.randperm(meta.shape[0]-1)
train_size = int(0.9 * meta.shape[0])
val_size = meta.shape[0] - train_size
train_indices = indices[:train_size]
val_indices = indices[train_size:]


num_epochs = 15000
batch_size = 25
for epoch in range(num_epochs):
    # t = random.sample(range(0, meta.shape[0] - 1), batch_size)
    t = torch.randperm(train_indices.numel())[:batch_size]
    t_1 = [s.item() + 1 for s in t]

    bloss = []
    torch.set_grad_enabled(True)
    preds_train = model(meta[t], coords[t], velocities[t], accelerations[t], forces[t])
    target_train = torch.cat([coords[t_1], velocities[t_1], accelerations[t_1], forces[t_1]], dim=1)

    loss = criterion(preds_train, target_train)
    bloss.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # optimizer.zero_grad()
    # bloss = torch.tensor(sum(bloss) / batch_size, requires_grad=True)
    # bloss.backward()
    # optimizer.step()

    # Print progress
    if (epoch) % 100 == 0:
        with torch.set_grad_enabled(False):
            tval = val_indices
            #tval = random.sample(range(0, meta.shape[0] - 1), batch_size)
            tval_1 = [s.item() + 1 for s in tval]
            bbloss = []
            preds_test = model(meta[tval], coords[tval], velocities[tval], accelerations[tval],
                          forces[tval])
            target_test = torch.cat([coords[tval_1], velocities[tval_1], accelerations[tval_1], forces[tval_1]], dim=1)
            loss_val = criterion(preds_test, target_test)
            print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {loss.item():.5f}, Validation Loss: {loss_val:.5f}')

model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
