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
    process_pdb_and_generate_animations, load_multimodel_pdb_coords, compute_rmsd, parse_pdb_ca_and_cb

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

model.cpu()
del model
gc.collect()
torch.cuda.empty_cache()


from pathlib import Path
import itertools, shutil, numpy as np, matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

FS_PER_UNRES_UNIT = 48.89
def read_frames(xfile):
    with Path(xfile).open(encoding="utf-8") as fh:
        while True:
            for header in fh:
                if not header or not header.lstrip().startswith('#'):
                    break
            else:
                return
            t_unres=float(header.split()[0]); time_fs=t_unres*FS_PER_UNRES_UNIT
            for line in fh:
                if not line or line.strip()=="#coordinates":
                    break
            coords=[]
            for line in fh:
                if line.lstrip().startswith('#'):
                    break
                coords.extend(float(tok) for tok in line.split())
            if not coords:
                return
            xyz=np.asarray(coords,dtype=np.float32).reshape(-1,3)
            yield time_fs,xyz

def split_atoms(xyz,n_res=46):
    ca=xyz[:n_res]; sc_all=xyz[n_res:]
    if sc_all.size==0:
        return ca,np.empty((0,3),dtype=xyz.dtype)
    dup=np.linalg.norm(sc_all[:,None]-ca[None],axis=2).min(1)<0.01
    return ca,sc_all[~dup]

def _pair_idx(ca,sc):
    if sc.size==0:
        return np.array([],dtype=int)
    return np.linalg.norm(sc[:,None]-ca[None],axis=2).argmin(1)

def show_frame(xyz,*,title="snapshot",ax=None,n_res=46):
    ca,sc=split_atoms(xyz,n_res); idx=_pair_idx(ca,sc)
    if ax is None:
        fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection="3d")
    ax.plot(ca[:,0],ca[:,1],ca[:,2],lw=1.8,color='k')
    for i,j in enumerate(idx):
        ax.plot([ca[j,0],sc[i,0]],[ca[j,1],sc[i,1]],[ca[j,2],sc[i,2]],lw=1.2,color='gold')
    ax.scatter(ca[:,0],ca[:,1],ca[:,2],s=22,c='dodgerblue')
    if sc.size:
        ax.scatter(sc[:,0],sc[:,1],sc[:,2],s=12,c='tomato')
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("x [Å]"); ax.set_ylabel("y [Å]"); ax.set_zlabel("z [Å]")
    ax.set_title(title)
    return ax

def animate_trajectory(xfile,*,n_res=46,stride=10,fps=15,slow=10,outfile="proteinA.mp4"):
    frames=list(itertools.islice(read_frames(xfile),None))[::stride]
    if not frames:
        raise RuntimeError(f"No frames in {xfile}")
    fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection="3d")
    ca0,sc0=split_atoms(frames[0][1],n_res); idx0=_pair_idx(ca0,sc0)
    line_ca,=ax.plot(ca0[:,0],ca0[:,1],ca0[:,2],lw=1.8,color='k')
    lines_cb=[ax.plot([ca0[j,0],sc0[i,0]],[ca0[j,1],sc0[i,1]],[ca0[j,2],sc0[i,2]],
                      lw=1.2,color='gold')[0] for i,j in enumerate(idx0)]
    scat_ca=ax.scatter(ca0[:,0],ca0[:,1],ca0[:,2],s=22,c='dodgerblue')
    scat_sc=ax.scatter(sc0[:,0],sc0[:,1],sc0[:,2],s=12,c='tomato')
    ax.set_box_aspect((1,1,1))
    ax.set_xlabel("x [Å]"); ax.set_ylabel("y [Å]"); ax.set_zlabel("z [Å]")
    def _update(i):
        _,xyz=frames[i]; ca,sc=split_atoms(xyz,n_res); idx=_pair_idx(ca,sc)
        line_ca.set_data(ca[:,0],ca[:,1]); line_ca.set_3d_properties(ca[:,2])
        for k,(l,j) in enumerate(zip(lines_cb,idx)):
            l.set_data([ca[j,0],sc[k,0]],[ca[j,1],sc[k,1]])
            l.set_3d_properties([ca[j,2],sc[k,2]])
        scat_ca._offsets3d=(ca[:,0],ca[:,1],ca[:,2])
        scat_sc._offsets3d=(sc[:,0],sc[:,1],sc[:,2])
        ax.set_title(f"t = {frames[i][0]/1000:.1f} ps")
        return (line_ca,*lines_cb,scat_ca,scat_sc)
    ani=FuncAnimation(fig,_update,frames=len(frames),
                      interval=int(1000/fps*slow),blit=False)
    if outfile:
        if shutil.which("ffmpeg") and outfile.lower().endswith(".mp4"):
            ani.save(outfile,writer=FFMpegWriter(fps=fps,codec="libx264"),dpi=500)
        else:
            gif=Path(outfile).with_suffix(".gif")
            ani.save(gif,writer=PillowWriter(fps=fps),dpi=500); outfile=str(gif)
        print("Saved movie to:",Path(outfile).resolve())
    return ani

X_FILE=r"proteinA/prota_MD_NVE-noxdr_MD000.x"
plt.style.use('dark_background')
# time_fs,coords=next(read_frames(X_FILE))
# show_frame(coords,title=f"t={time_fs/1000:.1f} ps"); plt.tight_layout(); plt.show()
animate_trajectory(X_FILE,stride=20,fps=5,slow=100,outfile="proteinA_movie.mp4")
plt.show()


