import os
import gc
import os
import random
import struct
import sys
import sysconfig
import time

import cv2
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt, animation
from matplotlib.animation import PillowWriter, FuncAnimation
from scipy.ndimage import gaussian_filter
from torch import nn, optim
from AVATAR import AvatarUNRES
from utils import switch_order, generate_volumetric_data, project3d_to_2d, create_2d_views, project_2d_to_3d, \
    process_pdb_and_generate_animations, load_multimodel_pdb_coords, compute_rmsd

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

pos2 = switch_order(pos)
vel = switch_order(vel)
acc = switch_order(acc)
force = switch_order(force)

meta = torch.tensor(meta_info).reshape(1000, 9 * 32)
coords = torch.tensor(pos)
coords2 = torch.tensor(pos2)
velocities = torch.tensor(vel)
accelerations = torch.tensor(acc)
forces = torch.tensor(force)

# scale = 1.
# offset = 0.
# outmap_min, _ = torch.min(coords, dim=1, keepdim=True)
# outmap_max, _ = torch.max(coords, dim=1, keepdim=True)
# coords = scale * (((coords - outmap_min) / (outmap_max - outmap_min)) + offset)
# outmap_min, _ = torch.min(velocities, dim=1, keepdim=True)
# outmap_max, _ = torch.max(velocities, dim=1, keepdim=True)
# velocities = scale * (((velocities - outmap_min) / (outmap_max - outmap_min)) + offset)
# outmap_min, _ = torch.min(accelerations, dim=1, keepdim=True)
# outmap_max, _ = torch.max(accelerations, dim=1, keepdim=True)
# accelerations = scale * (((accelerations - outmap_min) / (outmap_max - outmap_min)) + offset)
# outmap_min, _ = torch.min(forces, dim=1, keepdim=True)
# outmap_max, _ = torch.max(forces, dim=1, keepdim=True)
# forces = scale * (((forces - outmap_min) / (outmap_max - outmap_min)) + offset)

# coords_spaceXtime, velocities_spaceXtime, accelerations_spaceXtime, forces_spaceXtime = generate_volumetric_data(
# coords, velocities, accelerations, forces, grid_step, grid_padding, device)

ca_number = 46
x_ca = coords[:, 0:ca_number]
y_ca = coords[:, ca_number:ca_number*2]
z_ca = coords[:, ca_number*2:ca_number*3]
x_cb = coords[:, ca_number*3:ca_number*4]
y_cb = coords[:, ca_number*4:ca_number*5]
z_cb = coords[:,ca_number*5:ca_number*6]

x = torch.cat([x_ca,x_cb],dim=-1)
y = torch.cat([y_ca,y_cb],dim=-1)
z = torch.cat([z_ca,z_cb],dim=-1)
print(x.shape, y.shape, z.shape,coords2.shape)
coords3d = torch.tensor(np.stack((x, y, z), axis=-1), device=device)

xv = velocities[:, 0::3]
yv = velocities[:, 1::3]
zv = velocities[:, 2::3]
velocities3d = torch.tensor(np.stack((xv, yv, zv), axis=-1), device=device)
xa = accelerations[:, 0::3]
ya = accelerations[:, 1::3]
za = accelerations[:, 2::3]
accelerations3d = torch.tensor(np.stack((xa, ya, za), axis=-1), device=device)
xf = forces[:, 0::3]
yf = forces[:, 1::3]
zf = forces[:, 2::3]
forces3d = torch.tensor(np.stack((xf, yf, zf), axis=-1), device=device)

train_size = int(0.9 * meta.shape[0])
meta, meta_test = meta[:train_size], meta[train_size:]
coords3d, coords_test3d = coords3d[:train_size], coords3d[train_size:]
velocities3d, velocities_test3d = velocities3d[:train_size], velocities3d[train_size:]
accelerations3d, accelerations_test3d = accelerations3d[:train_size], accelerations3d[train_size:]
forces3d, forces_test3d = forces3d[:train_size], forces3d[train_size:]

# Training loop
indices = torch.randperm(meta.shape[0] - 1)

train_sizev2 = int(0.995 * meta.shape[0])
val_size = meta.shape[0] - train_sizev2
train_indices = indices[:train_sizev2]
val_indices = indices[train_sizev2:]

num_epochs = 10
batch_size = 2
bloss = []
bbloss = []
model = AvatarUNRES(meta, coords3d, velocities3d, accelerations3d, forces3d).to(
    device)
model.device = device
criterion = nn.MSELoss(reduction='mean')
# criterion = nn.HuberLoss(reduction='mean', delta=1.0)
# criterion = torch.nn.BCELoss()
lr = 1e-4
lr_mod = 1
# if os.path.exists(model_path):
#     model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, weight_decay=1e-4)
loss_idx = 0
start = time.time()
# for epoch in range(num_epochs):
#
#     torch.set_grad_enabled(True)
#     model.train()
#     model.batch_size = batch_size
#     # for param_group in optimizer.param_groups:
#     #     param_group['lr'] = lr
#
#     # RANDOM POINTS DYNAMIC dt LEARNING WITH STEP SIZE 1
#     # t = random.sample(range(0, meta.shape[0] - 1), batch_size)
#     t = torch.randperm(train_indices.numel())[:batch_size]
#     t_1 = [s.item() + 1 for s in t]
#     c_train, v_train, a_train, f_train, c_tz, v_tz, a_tz, f_tz = model(meta[t], coords3d[t], velocities3d[t],
#                                                                        accelerations3d[t], forces3d[t])
#
#     c_train = project_2d_to_3d(c_train, c_tz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
#                                model.grid_step, model.grid_padding, model.device)
#     v_train = project_2d_to_3d(v_train, v_tz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
#                                model.grid_step, model.grid_padding, model.device)
#     a_train = project_2d_to_3d(a_train, a_tz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
#                                model.grid_step, model.grid_padding, model.device)
#     f_train = project_2d_to_3d(f_train, f_tz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
#                                model.grid_step, model.grid_padding, model.device)
#
#     preds_train = torch.cat([c_train, v_train, a_train, f_train], dim=1)
#     # p, pz = create_2d_views(coords3d[t_1], model.grid_step, model.grid_padding, model.dist_coef, model.rot_ang,
#     #                         model.translation,
#     #                         model.camera_params,
#     #                         model.device)
#     # v, vz = create_2d_views(velocities3d[t_1], model.grid_step, model.grid_padding, model.dist_coef, model.rot_ang,
#     #                         model.translation,
#     #                         model.camera_params,
#     #                         model.device)
#     # a, az = create_2d_views(accelerations3d[t_1], model.grid_step, model.grid_padding, model.dist_coef, model.rot_ang,
#     #                         model.translation,
#     #                         model.camera_params,
#     #                         model.device)
#     # f, fz = create_2d_views(forces3d[t_1], model.grid_step, model.grid_padding, model.dist_coef, model.rot_ang,
#     #                         model.translation,
#     #                         model.camera_params,
#     #                         model.device)
#
#     target_train = torch.cat([coords3d[t_1], velocities3d[t_1], accelerations3d[t_1], forces3d[t_1]], dim=1)
#     # RANDOM POINTS DYNAMIC dt LEARNING WITH STEP SIZE 1
#     # print(preds_train.shape, target_train.shape)
#     loss = criterion(preds_train, target_train)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#
#     # Print progress
#     if epoch % 1 == 0:
#         with torch.set_grad_enabled(False):
#             model.eval()
#             tval = val_indices
#
#             # tval = random.sample(range(0, meta.shape[0] - 1), batch_size)
#             tval_1 = torch.tensor([s.item() + 1 for s in tval])
#
#             model.batch_size = coords3d[tval].shape[0]
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr
#
#             c_test, v_test, a_test, f_test, c_tz, v_tz, a_tz, f_tz = model(meta[tval], coords3d[tval],
#                                                                            velocities3d[tval],
#                                                                            accelerations3d[tval],
#                                                                            forces3d[tval])
#             preds_test = torch.cat([c_test, v_test, a_test, f_test], dim=1)
#
#             pt, ptz = create_2d_views(coords3d[tval_1], model.grid_step, model.grid_padding, model.dist_coef,
#                                       model.rot_ang,
#                                       model.translation,
#                                       model.camera_params,
#                                       model.device)
#             vt, vtz = create_2d_views(velocities3d[tval_1], model.grid_step, model.grid_padding, model.dist_coef,
#                                       model.rot_ang,
#                                       model.translation,
#                                       model.camera_params,
#                                       model.device)
#             at, atz = create_2d_views(accelerations3d[tval_1], model.grid_step, model.grid_padding, model.dist_coef,
#                                       model.rot_ang,
#                                       model.translation,
#                                       model.camera_params,
#                                       model.device)
#             ft, ftz = create_2d_views(forces3d[tval_1], model.grid_step, model.grid_padding, model.dist_coef,
#                                       model.rot_ang,
#                                       model.translation,
#                                       model.camera_params,
#                                       model.device)
#
#             target_test = torch.cat([pt, vt, at, ft], dim=1)
#             loss_val = criterion(preds_test, target_test)
#             bloss.append(loss.item())
#             bbloss.append(loss_val.item())
#             if epoch > 20 and loss_val < max(bbloss) and loss < max(bloss):
#                 torch.save(model.state_dict(), model_path)
#                 # batch_size += 1
#                 # if batch_size > 100:
#                 #     batch_size -= 1
#             sys.stdout.write("\rEpoch : %f" % round(epoch + 1))
#             sys.stdout.write("/ : %f" % round(num_epochs))
#             sys.stdout.write(" Train Loss : %f" % (round(loss.item(), 2)))
#             sys.stdout.write(" Validation Loss : %f" % (round(loss_val.item(), 2)))
#             sys.stdout.flush()

end = time.time()
print('\nLearning Time : ', round(end - start, 2), ' [s]')

# fig = plt.figure()
# # plt.style.use('dark_background')
# plt.plot(bloss, c='blue')
# plt.plot(bbloss, c='orange')
# plt.grid()
# plt.show()

p, v, a, f = torch.unsqueeze(coords_test3d[0], dim=0), torch.unsqueeze(velocities_test3d[0], dim=0), torch.unsqueeze(
    accelerations_test3d[0], dim=0), torch.unsqueeze(forces_test3d[0], dim=0)

pred_dynamics_pos = []
pred_dynamics_vel = []
pred_dynamics_acc = []
pred_dynamics_force = []

gtdlen = int(meta_test.shape[0] * 0.1)

model.batch_size = 1
for param_group in optimizer.param_groups:
    param_group['lr'] = lr * lr_mod
model.eval()
start = time.time()
with torch.set_grad_enabled(False):
    for i in range(int(gtdlen)):
        # p, v, a, f, pz, vz, az, fz = model(meta_test[i], p, v, a, f)


        cprim, vprim, aprim, fprim = torch.unsqueeze(coords_test3d[i], dim=0), torch.unsqueeze(velocities_test3d[i],
                                                                                               dim=0), torch.unsqueeze(
            accelerations_test3d[i], dim=0), torch.unsqueeze(forces_test3d[i], dim=0)

        pt, ptz = create_2d_views(cprim, model.grid_step, model.grid_padding, model.dist_coef,
                                model.rot_ang,
                                model.translation,
                                model.camera_params,
                                model.device)

        vt, vtz = create_2d_views(vprim, model.grid_step, model.grid_padding, model.dist_coef,
                                model.rot_ang,
                                model.translation,
                                model.camera_params,
                                model.device)
        at, atz = create_2d_views(aprim, model.grid_step, model.grid_padding, model.dist_coef,
                                model.rot_ang,
                                model.translation,
                                model.camera_params,
                                model.device)
        ft, ftz = create_2d_views(fprim, model.grid_step, model.grid_padding, model.dist_coef,
                                model.rot_ang,
                                model.translation,
                                model.camera_params,
                                model.device)

        p = project_2d_to_3d(pt, ptz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
                             model.grid_step, model.grid_padding, model.device)
        # p = p.unsqueeze(0)

        v = project_2d_to_3d(vt, vtz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
                             model.grid_step, model.grid_padding, model.device)
        # v = v.unsqueeze(0)

        a = project_2d_to_3d(at, atz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
                             model.grid_step, model.grid_padding, model.device)
        # a = a.unsqueeze(0)

        f = project_2d_to_3d(ft, ftz, model.dist_coef, model.rot_ang, model.translation, model.camera_params,
                             model.grid_step, model.grid_padding, model.device)
        # f = f.unsqueeze(0)

        # c = torch.unsqueeze(c[0, :, :], dim=0)
        # print(torch.max(c),torch.min(c))
        pred_dynamics_pos.append(p.cpu().numpy())
        pred_dynamics_vel.append(v.cpu().numpy())
        pred_dynamics_acc.append(a.cpu().numpy())
        pred_dynamics_force.append(f.cpu().numpy())

end = time.time()
print('Sequence generation speed :', int(gtdlen) / round((end - start), 4), ' [fps]')  # speed
pred_dynamics_pos = np.array(pred_dynamics_pos)
pred_dynamics_vel = np.array(pred_dynamics_vel)
pred_dynamics_acc = np.array(pred_dynamics_acc)
pred_dynamics_force = np.array(pred_dynamics_force)
ground_truth_dynamics_pos = []
ground_truth_dynamics_vel = []
ground_truth_dynamics_acc = []
ground_truth_dynamics_force = []

for i in range(int(gtdlen)):
    ground_truth_dynamics_pos.append(coords_test3d[i].cpu())
    ground_truth_dynamics_vel.append(velocities_test3d[i].cpu())
    ground_truth_dynamics_acc.append(accelerations_test3d[i].cpu())
    ground_truth_dynamics_force.append(forces_test3d[i].cpu())

ground_truth_dynamics_pos = np.array(ground_truth_dynamics_pos)
ground_truth_dynamics_vel = np.array(ground_truth_dynamics_vel)
ground_truth_dynamics_acc = np.array(ground_truth_dynamics_acc)
ground_truth_dynamics_force = np.array(ground_truth_dynamics_force)

pdb_path = "proteinA/prota_nmr.pdb"
process_pdb_and_generate_animations(
    pdb_path,
    pred_dynamics_pos,
    ground_truth_dynamics_pos,
    output_folder="animation_pdbs",
    separate_frames=False
)
gt_coords = load_multimodel_pdb_coords("animation_pdbs/ground_truth_trajectory.pdb")
pred_coords = load_multimodel_pdb_coords("animation_pdbs/predicted_trajectory.pdb")

fig = plt.figure()
plt.style.use('dark_background')
prota = fig.add_subplot(111, projection='3d')
prota.set_xlabel("x")
prota.set_ylabel("y")
prota.set_zlabel("z")
marker_size = 1.5
linewidth = 1
alpha = 0.9

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

show_res_text = False
show_res_markers = True

def parse_pdb_ca_and_cb(pdb_file):
    d = {}
    order = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                rn = line[22:26].strip()
                key = rn
                if key not in d:
                    d[key] = {"CA": None, "CB": None}
                    order.append(key)
                name = line[13:15]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                if name == "CA":
                    d[key]["CA"] = [x, y, z]
                elif name == "CB":
                    d[key]["CB"] = [x, y, z]
    ca_coords = []
    cb_coords = []
    for k in order:
        ca_coords.append(d[k]["CA"])
        cb_coords.append(d[k]["CB"] if d[k]["CB"] else [np.nan, np.nan, np.nan])
    ca_coords = np.array(ca_coords)
    cb_coords = np.array(cb_coords)
    edges_ca = []
    for i in range(len(ca_coords)-1):
        edges_ca.append((i, i+1))
    edges_cb = []
    for i in range(len(ca_coords)):
        if not np.isnan(cb_coords[i]).any():
            edges_cb.append((i, i))
    return ca_coords, cb_coords, edges_ca, edges_cb, order

pdb_filename = "proteinA/prota_nmr.pdb"
ca_coords, cb_coords, edges_ca, edges_cb, residues_order = parse_pdb_ca_and_cb(pdb_filename)

N = len(ca_coords)

ground_truth_dynamics_ca = []
ground_truth_dynamics_cb = []
pred_dynamics_ca = []
pred_dynamics_cb = []
for i in range(gtdlen):
    shift_gt = np.random.normal(scale=0.3, size=(N,3))*(i/gtdlen)
    shift_pr = np.random.normal(scale=0.3, size=(N,3))*(i/gtdlen) + 1.0
    gca = ca_coords.copy()+shift_gt
    gcb = cb_coords.copy()+shift_gt
    pca = ca_coords.copy()+shift_pr
    pcb = cb_coords.copy()+shift_pr
    ground_truth_dynamics_ca.append(gca)
    ground_truth_dynamics_cb.append(gcb)
    pred_dynamics_ca.append(pca)
    pred_dynamics_cb.append(pcb)

# all_gt = np.concatenate(ground_truth_dynamics_ca+ground_truth_dynamics_cb, axis=0)
all_pr = np.concatenate(pred_dynamics_ca+pred_dynamics_cb, axis=0)
all_gt = np.concatenate(ground_truth_dynamics_pos, axis=0)
# all_pr = np.concatenate(pred_dynamics_pos, axis=0)

all_data = np.vstack([all_gt, all_pr])
x_min, x_max = np.nanmin(all_data[:,0]), np.nanmax(all_data[:,0])
y_min, y_max = np.nanmin(all_data[:,1]), np.nanmax(all_data[:,1])
z_min, z_max = np.nanmin(all_data[:,2]), np.nanmax(all_data[:,2])
margin = 0.5

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d(x_min - margin, x_max + margin)
ax.set_ylim3d(y_min - margin, y_max + margin)
ax.set_zlim3d(z_min - margin, z_max + margin)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

gt_scatter_ca = ax.scatter([], [], [], c='b', alpha=0.6)
pred_scatter_ca = ax.scatter([], [], [], c='r', alpha=0.3)
gt_scatter_cb = ax.scatter([], [], [], marker='o', c='b', alpha=0.6)
pred_scatter_cb = ax.scatter([], [], [], marker='o', c='r', alpha=0.3)

gt_lines_ca = []
pred_lines_ca = []
for _ in edges_ca:
    l1, = ax.plot([], [], [], c='b', alpha=0.4)
    l2, = ax.plot([], [], [], c='r', alpha=0.4)
    gt_lines_ca.append(l1)
    pred_lines_ca.append(l2)

gt_lines_cb = []
pred_lines_cb = []
for _ in edges_cb:
    l1, = ax.plot([], [], [], c='b', alpha=0.6)
    l2, = ax.plot([], [], [], c='r', alpha=0.6)
    gt_lines_cb.append(l1)
    pred_lines_cb.append(l2)

res_texts = []
if show_res_text:
    for _ in residues_order:
        txt = ax.text(0, 0, 0, "", color='white', fontsize=8)
        res_texts.append(txt)

def init():
    gt_scatter_ca._offsets3d = ([], [], [])
    pred_scatter_ca._offsets3d = ([], [], [])
    gt_scatter_cb._offsets3d = ([], [], [])
    pred_scatter_cb._offsets3d = ([], [], [])
    for ln in gt_lines_ca+pred_lines_ca+gt_lines_cb+pred_lines_cb:
        ln.set_data([], [])
        ln.set_3d_properties([])
    if show_res_text:
        for t in res_texts:
            t.set_position((0, 0))
            t.set_3d_properties(0, zdir='z')
            t.set_text("")
    return [gt_scatter_ca, pred_scatter_ca, gt_scatter_cb, pred_scatter_cb]+gt_lines_ca+pred_lines_ca+gt_lines_cb+pred_lines_cb+(res_texts if show_res_text else [])


def update(frame):
    gca = ground_truth_dynamics_pos[frame,:46]
    gcb = ground_truth_dynamics_pos[frame,46:]
    pca = pred_dynamics_ca[frame]
    pcb = pred_dynamics_cb[frame]
    gt_scatter_ca._offsets3d = (gca[:,0], gca[:,1], gca[:,2])
    pred_scatter_ca._offsets3d = (pca[:,0], pca[:,1], pca[:,2])
    gt_scatter_cb._offsets3d = (gcb[:,0], gcb[:,1], gcb[:,2])
    pred_scatter_cb._offsets3d = (pcb[:,0], pcb[:,1], pcb[:,2])
    for i, (a,b) in enumerate(edges_ca):
        gt_lines_ca[i].set_data([gca[a,0], gca[b,0]], [gca[a,1], gca[b,1]])
        gt_lines_ca[i].set_3d_properties([gca[a,2], gca[b,2]])
        pred_lines_ca[i].set_data([pca[a,0], pca[b,0]], [pca[a,1], pca[b,1]])
        pred_lines_ca[i].set_3d_properties([pca[a,2], pca[b,2]])
    for i, (a,b) in enumerate(edges_cb):
        gt_lines_cb[i].set_data([gca[a,0], gcb[b,0]], [gca[a,1], gcb[b,1]])
        gt_lines_cb[i].set_3d_properties([gca[a,2], gcb[b,2]])
        pred_lines_cb[i].set_data([pca[a,0], pcb[b,0]], [pca[a,1], pcb[b,1]])
        pred_lines_cb[i].set_3d_properties([pca[a,2], pcb[b,2]])
    if show_res_text:
        for i, t in enumerate(res_texts):
            if not np.isnan(gcb[i]).any():
                x = gcb[i,0]
                y = gcb[i,1]
                z = gcb[i,2]
                t.set_position((x, y))
                t.set_3d_properties(z, zdir='z')
                t.set_text(residues_order[i])
    return [gt_scatter_ca, pred_scatter_ca, gt_scatter_cb, pred_scatter_cb]+gt_lines_ca+pred_lines_ca+gt_lines_cb+pred_lines_cb+(res_texts if show_res_text else [])

anim = FuncAnimation(fig, update, frames=gtdlen, init_func=init, interval=250, blit=False, repeat=True)
anim.save("proteinA-folding_testing.gif", dpi=300, writer=PillowWriter(fps=10))
plt.show()



model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()
