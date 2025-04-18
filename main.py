import gc
import os
import random
import sys
import sysconfig
import time
import matplotlib
import numpy as np
import torch
from torch import nn, optim
from AVATAR import AvatarUNRES

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


from pathlib import Path
import itertools, shutil, numpy as np, matplotlib.pyplot as plt, torch
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from mpl_toolkits.mplot3d import Axes3D

FS_PER_UNRES_UNIT = 48.89

def _grab_block(fh):
    data = []
    for line in fh:
        if line.lstrip().startswith('#'):
            return data, line
        data.extend(map(float, line.split()))
    return data, ''                     # EOF


def _skip_to_tag(fh, keyword):
    for line in fh:
        if line.lstrip().startswith('#') and keyword in line:
            return line
    return ''                           # not found


def _ensure_tag(fh, current_tag, keyword):
    if current_tag and current_tag.lstrip().startswith('#') and keyword in current_tag:
        return current_tag
    return _skip_to_tag(fh, keyword)


def _fit_len(arr, L, fill=float('nan')):
    if len(arr) < L:
        return arr + [fill] * (L - len(arr))
    if len(arr) > L:
        return arr[:L]
    return arr


def load_trajectory_tensors(xfile: str):
    times, frc = [], []
    signal = [],[],[],[]
    n_atoms = None
    var_names = ['#coordinates','#velocities','#accelerations','#potential forces']
    for k in range(0, 4):
        with Path(xfile).open(encoding="utf-8") as fh:
            tag = ''
            t_unres = 0
            while True:
                tag = _skip_to_tag(fh, var_names[k])
                if not tag:
                    break
                x, tag = _grab_block(fh)
                if t_unres == 999:
                    print(x)
                cur_atoms = len(x) // 3
                if n_atoms is None:
                    n_atoms = cur_atoms
                x = _fit_len(x, n_atoms * 3)
                signal[k].append(x)
                if k == 0:
                    times.append(t_unres * FS_PER_UNRES_UNIT)
                t_unres+=1
    pos,vel,acc,frc = signal[0], signal[1], signal[2], signal[3]
    if not pos:
        raise RuntimeError("no coordinate blocks found in the file")
    n_frames = len(pos)

    def _mk(arr):
        return torch.tensor(np.asarray(arr, dtype=np.float32).reshape(n_frames, n_atoms, 3))

    return (torch.tensor(np.asarray(times, dtype=np.float32)),
            _mk(pos), _mk(vel), _mk(acc), _mk(frc))


def _to_tensors(times, pos, vel, acc, frc):
    if not pos:
        raise RuntimeError("no complete frames found in the file")

    n_frames = len(pos)
    n_atoms  = len(pos[0]) // 3

    def _mk(arr):
        return torch.tensor(np.asarray(arr, dtype=np.float32)
                            .reshape(n_frames, n_atoms, 3))
    return (torch.tensor(np.asarray(times, dtype=np.float32)),
            _mk(pos), _mk(vel), _mk(acc), _mk(frc))

def read_frames(xfile):
    with Path(xfile).open(encoding="utf-8") as fh:
        while True:
            for header in fh:
                if not header or not header.lstrip().startswith('#'):
                    break
            else:
                return
            t_unres = float(header.split()[0])
            time_fs = t_unres * FS_PER_UNRES_UNIT
            for line in fh:
                if not line or line.strip() == "#coordinates":
                    break
            coords = []
            for line in fh:
                if line.lstrip().startswith('#'):
                    break
                coords.extend(float(tok) for tok in line.split())
            if not coords:
                return
            xyz = np.asarray(coords, dtype=np.float32).reshape(-1, 3)
            yield time_fs, xyz

def split_atoms(xyz, n_res=46):
    ca = xyz[:n_res]
    sc_all = xyz[n_res:]
    if sc_all.size == 0:
        return ca, np.empty((0, 3), dtype=xyz.dtype)
    dup = np.linalg.norm(sc_all[:, None] - ca[None], axis=2).min(1) < 0.01
    return ca, sc_all[~dup]

def _pair_idx(ca, sc):
    if sc.size == 0:
        return np.array([], dtype=int)
    return np.linalg.norm(sc[:, None] - ca[None], axis=2).argmin(1)

def animate_trajectory(xfile, *, n_res=46, stride=10,
                       fps=15, slow=10, outfile="proteinA.mp4"):
    frames = list(itertools.islice(read_frames(xfile), None))[::stride]
    if not frames:
        raise RuntimeError(f"No frames in {xfile}")

    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection="3d")

    ca0, sc0 = split_atoms(frames[0][1], n_res)
    idx0     = _pair_idx(ca0, sc0)

    line_ca, = ax.plot(ca0[:, 0], ca0[:, 1], ca0[:, 2], lw=1.8, color='k')
    lines_cb = [ax.plot([ca0[j, 0], sc0[i, 0]],
                        [ca0[j, 1], sc0[i, 1]],
                        [ca0[j, 2], sc0[i, 2]],
                        lw=1.2, color='gold')[0] for i, j in enumerate(idx0)]
    scat_ca  = ax.scatter(ca0[:, 0], ca0[:, 1], ca0[:, 2], s=22, c='dodgerblue')
    scat_sc  = ax.scatter(sc0[:, 0], sc0[:, 1], sc0[:, 2], s=12, c='tomato')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [Å]"); ax.set_ylabel("y [Å]"); ax.set_zlabel("z [Å]")

    def _update(i):
        _, xyz = frames[i]
        ca, sc = split_atoms(xyz, n_res)
        idx    = _pair_idx(ca, sc)

        line_ca.set_data(ca[:, 0], ca[:, 1]); line_ca.set_3d_properties(ca[:, 2])
        for k, (l, j) in enumerate(zip(lines_cb, idx)):
            l.set_data([ca[j, 0], sc[k, 0]], [ca[j, 1], sc[k, 1]])
            l.set_3d_properties([ca[j, 2], sc[k, 2]])

        scat_ca._offsets3d = (ca[:, 0], ca[:, 1], ca[:, 2])
        scat_sc._offsets3d = (sc[:, 0], sc[:, 1], sc[:, 2])

        ax.set_title(f"t = {frames[i][0] / 1000:.1f} ps")
        return (line_ca, *lines_cb, scat_ca, scat_sc)

    ani = FuncAnimation(fig, _update, frames=len(frames),
                        interval=int(1000 / fps * slow), blit=False)
    if outfile:
        if shutil.which("ffmpeg") and outfile.lower().endswith(".mp4"):
            ani.save(outfile, writer=FFMpegWriter(fps=fps, codec="libx264"),
                     dpi=500)
        else:
            gif = Path(outfile).with_suffix(".gif")
            ani.save(gif, writer=PillowWriter(fps=fps), dpi=500)
            outfile = str(gif)
        print("Saved movie to:", Path(outfile).resolve())
    return ani

X_FILE = r"proteinA/prota_MD_NVE-noxdr_MD000.x"

times_t, pos_t, vel_t, acc_t, forces_t = load_trajectory_tensors(X_FILE)
times_t, pos_t, vel_t, acc_t, forces_t = times_t.to(device), pos_t.to(device), vel_t.to(device), acc_t.to(device), forces_t.to(device)
print("Loaded tensors shapes:",
      "\n  times  :", times_t.shape,
      "\n  pos    :", pos_t.shape,
      "\n  vel    :", vel_t.shape,
      "\n  acc    :", acc_t.shape,
      "\n  forces :", forces_t.shape)

pos_centroids = pos_t.mean(dim=1, keepdim=True)
vel_centroids = vel_t.mean(dim=1, keepdim=True)
acc_centroids = acc_t.mean(dim=1, keepdim=True)

pos_t = pos_t - pos_centroids
vel_t = vel_t - vel_centroids
acc_t = acc_t - acc_centroids
# plt.style.use('dark_background')
# anim = animate_trajectory(X_FILE, stride=20, fps=5,slow=100, outfile="proteinA_whole_dynamics.mp4")
# plt.show()

train_size = int(0.9 * pos_t.shape[0])
coords3d, coords_test3d = pos_t[:train_size], pos_t[train_size:]
velocities3d, velocities_test3d = vel_t[:train_size], vel_t[train_size:]
accelerations3d, accelerations_test3d = acc_t[:train_size], acc_t[train_size:]
forces3d, forces_test3d = forces_t[:train_size], forces_t[train_size:]

n_frames   = pos_t.shape[0]
train_size = int(0.90 * n_frames)

train_pool = torch.arange(0,          train_size-1, device=device)
val_pool   = torch.arange(train_size, n_frames-1,   device=device)

perm_train   = torch.randperm(train_pool.numel(), device=device)
perm_val     = torch.randperm(val_pool.numel(),   device=device)
train_cursor = val_cursor = 0                     #

model_path     = './avatar_unres_best.pt'
load_if_exists = False
skip_training = True

num_epochs       = 8000
batch_size       = 16
base_lr          = 1e-3
max_grad_norm    = 1.0
noise_cfg = dict(pos = 0.025,vel = 0.025,acc = 0.025)

model      = AvatarUNRES(pos_t, vel_t, acc_t).to(device)
optimiser   = optim.Adam(model.parameters(), lr=base_lr, weight_decay=1e-4,amsgrad=True)
scheduler   = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=250,min_lr=1e-6)
print('LR set:',scheduler.get_last_lr())
criterion   = nn.MSELoss()

best_val = float('inf')
best_ckpt = None

if skip_training:
    load_if_exists = True

if load_if_exists and os.path.isfile(model_path):
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    optimiser.load_state_dict(ckpt['optim'])
    best_val = ckpt['best_val']
    print(f'✓ loaded checkpoint  (val loss {best_val:.4e})')

bloss, vloss = [], []
start = time.time()
if skip_training:
    pass
else:
    # Note: ─────────────────────────── training loop ─────────────────────────
    for epoch in range(num_epochs):
        if train_cursor + batch_size > perm_train.numel():
            perm_train, train_cursor = torch.randperm(train_pool.numel(), device=device), 0
        idx  = perm_train[train_cursor : train_cursor+batch_size]
        t    = train_pool[idx]; train_cursor += batch_size

        # Note: ---------- data augmentation ----------------------------------
        pos_in = pos_t[t] + torch.randn_like(pos_t[t]) * noise_cfg['pos']
        vel_in = vel_t[t] + torch.randn_like(vel_t[t]) * noise_cfg['vel']
        acc_in = acc_t[t] + torch.randn_like(acc_t[t]) * noise_cfg['acc']

        # Attention: random sign‑flip (global mirror)
        if torch.rand(1) < 0.5:
            sign = (-1)**torch.randint(0,2,(1,),device=device)
            pos_in, vel_in, acc_in = pos_in*sign, vel_in*sign, acc_in*sign

        model.train()
        pr_pos, pr_vel, pr_acc = model(pos_in, vel_in, acc_in)
        pred   = torch.cat([pr_pos, pr_vel, pr_acc], dim=1)
        target = torch.cat([pos_t[t+1], vel_t[t+1], acc_t[t+1]], dim=1)

        loss = criterion(pred, target)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimiser.step()
        bloss.append(loss.item())

        # Note: --------------- validation ------------------------------------
        if val_cursor + batch_size > perm_val.numel():
            perm_val, val_cursor = torch.randperm(val_pool.numel(), device=device), 0
        vidx  = perm_val[val_cursor : val_cursor+batch_size]
        t_val = val_pool[vidx]; val_cursor += batch_size

        model.eval()
        with torch.no_grad():
            pv_pos, pv_vel, pv_acc = model(pos_t[t_val], vel_t[t_val], acc_t[t_val])
            pred_val   = torch.cat([pv_pos, pv_vel, pv_acc], dim=1)
            target_val = torch.cat([pos_t[t_val+1], vel_t[t_val+1], acc_t[t_val+1]], dim=1)
            v_loss = criterion(pred_val, target_val).item()
            vloss.append(v_loss)

        # ---------- LR scheduler & checkpoint --------------------------
        scheduler.step(v_loss)

        if v_loss < best_val:
            best_val = v_loss
            best_ckpt = {
                'state_dict': model.state_dict(),
                'optim'    : optimiser.state_dict(),
                'best_val' : best_val
            }
            torch.save(best_ckpt, model_path)
            flag = ' ⭑ saved'
        else:
            flag = ''

        if epoch % 50 == 0:
            lr_now = optimiser.param_groups[0]['lr']
            print(f'Epoch {epoch+1:4d}/{num_epochs} · '
                  f'lr {lr_now:.2e} · '
                  f'train L {loss.item():.3e} · val L {v_loss:.3e}{flag}')

    print(f'\n⌛ finished in {(time.time()-start)/60:.1f} min')
    fig = plt.figure()
    # plt.style.use('dark_background')
    plt.plot(bloss, c='blue', label='train_loss')
    plt.plot(vloss, c='orange', label='test_loss')
    plt.legend()
    plt.grid()
    plt.show()

if best_ckpt is not None:
    model.load_state_dict(best_ckpt['state_dict'])


plt.style.use('dark_background')
gtdlen   = int(coords_test3d.shape[0] * 0.30)
model.eval()
pred_frames = []
with torch.no_grad():
    for i in range(gtdlen):
        t0 = train_size + i
        p, _, _ = model(pos_t[t0:t0+1], vel_t[t0:t0+1], acc_t[t0:t0+1])
        pred_frames.append(p.squeeze(0).cpu().numpy())
gt_frames   = pos_t[train_size+1:train_size+1+gtdlen].cpu().numpy()
time_frames = times_t[train_size+1:train_size+1+gtdlen].cpu().numpy()

def _pair_idx(ca, sc):
    if sc.size == 0:
        return np.array([], dtype=int)
    return np.linalg.norm(sc[:, None] - ca[None], axis=2).argmin(1)

def animate_compare(gt, pr, times, n_res=46, fps=1, slow=8,
                    out='compare.mp4', show_cb=True):
    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection='3d')

    ca_g0, sc_g0 = split_atoms(gt[0], n_res)
    ca_p0, sc_p0 = split_atoms(pr[0], n_res)

    idx_g0 = _pair_idx(ca_g0, sc_g0)
    idx_p0 = _pair_idx(ca_p0, sc_p0)

    lg, = ax.plot(ca_g0[:, 0], ca_g0[:, 1], ca_g0[:, 2],
                  lw=1.8, c='dodgerblue')
    lp, = ax.plot(ca_p0[:, 0], ca_p0[:, 1], ca_p0[:, 2],
                  lw=1.8, c='orangered')

    lines_g = []
    lines_p = []
    if show_cb:
        for i, j in enumerate(idx_g0):
            lines_g.append(ax.plot([ca_g0[j, 0], sc_g0[i, 0]],
                                   [ca_g0[j, 1], sc_g0[i, 1]],
                                   [ca_g0[j, 2], sc_g0[i, 2]],
                                   lw=1.0, c='dodgerblue')[0])
        for i, j in enumerate(idx_p0):
            lines_p.append(ax.plot([ca_p0[j, 0], sc_p0[i, 0]],
                                   [ca_p0[j, 1], sc_p0[i, 1]],
                                   [ca_p0[j, 2], sc_p0[i, 2]],
                                   lw=1.0, c='orangered')[0])

    sg = ax.scatter(ca_g0[:, 0], ca_g0[:, 1], ca_g0[:, 2],
                    s=18, c='dodgerblue')
    sp = ax.scatter(ca_p0[:, 0], ca_p0[:, 1], ca_p0[:, 2],
                    s=18, c='orangered')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel('x [Å]')
    ax.set_ylabel('y [Å]')
    ax.set_zlabel('z [Å]')

    def up(i):
        ca_g, sc_g = split_atoms(gt[i], n_res)
        ca_p, sc_p = split_atoms(pr[i], n_res)
        lg.set_data(ca_g[:, 0], ca_g[:, 1])
        lg.set_3d_properties(ca_g[:, 2])
        lp.set_data(ca_p[:, 0], ca_p[:, 1])
        lp.set_3d_properties(ca_p[:, 2])
        sg._offsets3d = (ca_g[:, 0], ca_g[:, 1], ca_g[:, 2])
        sp._offsets3d = (ca_p[:, 0], ca_p[:, 1], ca_p[:, 2])

        if show_cb:
            idx_g = _pair_idx(ca_g, sc_g)
            idx_p = _pair_idx(ca_p, sc_p)
            for k, (l, j) in enumerate(zip(lines_g, idx_g)):
                l.set_data([ca_g[j, 0], sc_g[k, 0]],
                           [ca_g[j, 1], sc_g[k, 1]])
                l.set_3d_properties([ca_g[j, 2], sc_g[k, 2]])
            for k, (l, j) in enumerate(zip(lines_p, idx_p)):
                l.set_data([ca_p[j, 0], sc_p[k, 0]],
                           [ca_p[j, 1], sc_p[k, 1]])
                l.set_3d_properties([ca_p[j, 2], sc_p[k, 2]])

        ax.set_title(f't = {-times[i] / 1000:.1f} ps')
        return (lg, lp, sg, sp, *lines_g, *lines_p) if show_cb else (lg, lp, sg, sp)

    ani = FuncAnimation(fig, up, frames=len(gt),
                        interval=int(1000 / fps * slow), blit=False)

    if shutil.which('ffmpeg') and out.lower().endswith('.mp4'):
        ani.save(out, writer=FFMpegWriter(fps=fps, codec='libx264'), dpi=300)
    else:
        ani.save(Path(out).with_suffix('.gif'),
                 writer=PillowWriter(fps=fps), dpi=300)
    plt.show()

# animate_compare(gt_frames, pred_frames, time_frames,
#                 n_res=46, fps=2, slow=4, out='compare.mp4',
#                 show_cb=True)

# compute per‑frame RMSD on test set
model.eval()
rmsd_pos, rmsd_vel, rmsd_acc = [], [], []
with torch.no_grad():
    # test frames are train_size … n_frames‑2 (so we can compare t → t+1)
    for t in range(train_size, pos_t.shape[0] - 1):
        pr_pos, pr_vel, pr_acc = model(
            pos_t[t:t+1], vel_t[t:t+1], acc_t[t:t+1]
        )
        true_pos = pos_t[t+1]
        true_vel = vel_t[t+1]
        true_acc = acc_t[t+1]
        diff_pos = pr_pos.squeeze(0) - true_pos
        diff_vel = pr_vel.squeeze(0) - true_vel
        diff_acc = pr_acc.squeeze(0) - true_acc
        rmsd_pos.append(torch.sqrt((diff_pos**2).mean()).item())
        rmsd_vel.append(torch.sqrt((diff_vel**2).mean()).item())
        rmsd_acc.append(torch.sqrt((diff_acc**2).mean()).item())

# plot
# plt.figure()
# # plt.plot(rmsd_pos, label='AVG Position')
# # plt.plot(rmsd_vel, label='AVG Velocity')
# plt.plot(rmsd_acc, label='AVG Acceleration')
# plt.xlabel('Test frame index')
# plt.ylabel('RMSD [U]')
# plt.legend()
# plt.grid(True)
# plt.show()
n_atoms = gt_frames[0].shape[0]
half = n_atoms // 2

def rmsd_matrices(gt, pr):
    diff = pr[:, None, :] - gt[None, :, :]
    return np.abs(diff[:, :, 0]), np.abs(diff[:, :, 1]), np.abs(diff[:, :, 2])

mats_x, mats_y, mats_z = zip(*(rmsd_matrices(gt_frames[i], pred_frames[i]) for i in range(len(gt_frames))))

# swap top and bottom halves
mats_x = [np.vstack((m[half:], m[:half])) for m in mats_x]
mats_y = [np.vstack((m[half:], m[:half])) for m in mats_y]
mats_z = [np.vstack((m[half:], m[:half])) for m in mats_z]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
vmax = max(np.max(m) for m in mats_x + mats_y + mats_z)
im_x = axes[0].imshow(mats_x[0], vmin=0, vmax=vmax, origin='lower', cmap='inferno')
axes[0].set_title('X RMSD')
im_y = axes[1].imshow(mats_y[0], vmin=0, vmax=vmax, origin='lower', cmap='inferno')
axes[1].set_title('Y RMSD')
im_z = axes[2].imshow(mats_z[0], vmin=0, vmax=vmax, origin='lower', cmap='inferno')
axes[2].set_title('Z RMSD')
for ax in axes:
    ax.set_xlabel('true atom index')
    ax.set_ylabel('pred atom index')
cbar = fig.colorbar(im_z, ax=axes.ravel().tolist(), shrink=0.6)
cbar.set_label('Distance [Å]')

def update(i):
    im_x.set_data(mats_x[i])
    im_y.set_data(mats_y[i])
    im_z.set_data(mats_z[i])
    fig.suptitle(f'Frame {i}')
    return (im_x, im_y, im_z)

ani = FuncAnimation(fig, update, frames=len(gt_frames), interval=200)
plt.show()

gif_path = Path('rmsd_xyz_heatmap_swapped.gif')
ani.save(gif_path, writer=PillowWriter(fps=5))
