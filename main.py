import math
import random
import sys
import sysconfig

import matplotlib
import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import normalize

from AVATAR import AvatarUNRES
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

meta = torch.tensor(meta_info, device=device)
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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop


indices = torch.randperm(meta.shape[0])
train_size = int(0.9 * meta.shape[0])
val_size = meta.shape[0] - train_size
train_indices = indices[:train_size]
val_indices = indices[train_size:]

num_epochs = 6000
batch_size = 50
for epoch in range(num_epochs):
    t = random.sample(range(0, meta.shape[0] - 2), batch_size)
    # t_1 = [x + 1 for x in t]
    bloss = []
    torch.set_grad_enabled(True)
    for tstep in t:
        t_1 = tstep + 1
        preds = model(meta[tstep], coords[tstep], velocities[tstep], accelerations[tstep], forces[tstep])
        target = torch.cat([coords[t_1], velocities[t_1], accelerations[t_1], forces[t_1]])
        loss = criterion(preds, target)
        bloss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    optimizer.zero_grad()
    bloss = torch.tensor(sum(bloss) / batch_size, requires_grad=True)
    bloss.backward()
    optimizer.step()

    # Print progress
    if (epoch) % 100 == 0:
        with torch.set_grad_enabled(False):
            tval = random.sample(range(0, meta.shape[0] - 2), batch_size)
            bbloss = []
            loss = 0.
            for tvalstep in tval:
                tval_1 = tvalstep + 1
                preds = model(meta[tvalstep], coords[tvalstep], velocities[tvalstep], accelerations[tvalstep], forces[tvalstep])
                target = torch.cat([coords[tval_1], velocities[tval_1], accelerations[tval_1], forces[tval_1]])
                loss += criterion(preds, target)
            print(f'Epoch [{epoch + 1}/{num_epochs}],Train Loss: {bloss.item():.5f}, Validation Loss: {loss:.5f}')

