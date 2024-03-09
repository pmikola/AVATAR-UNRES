import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import relu


def switch_order(input):
    x_g = []
    y_g = []
    z_g = []
    for i in range(len(input)):
        xx = 0
        yy = 14
        zz = 29
        x, y, z, xp, yp, zp = [], [], [], [], [], []
        for j in range(0, int(len(input[0]) / 3)):
            # print(pred_dynamics_pos.shape)
            x.append(input[i][xx])
            y.append(input[i][yy])
            z.append(input[i][zz])

            xx += 1
            yy += 1
            zz += 1
            if zz % 15 == 0:
                xx += 15
                yy += 15
                zz += 15
        x_g.append(x)
        y_g.append(y)
        z_g.append(z)

    for i in range(len(input)):
        k = 0
        for j in range(0, int(len(input[0]) / 3)):
            input[i][k] = x_g[i][j]
            input[i][k + 1] = y_g[i][j]
            input[i][k + 2] = z_g[i][j]
            k += 3

    return input


def real_imaginary_relu(z):
    return relu(z.real) + 1.j * relu(z.imag)


def generate_volumetric_data(coords, velocities, accelerations, forces, grid_step, grid_padding, device):
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

    return coords_spaceXtime, velocities_spaceXtime, accelerations_spaceXtime, forces_spaceXtime


def project3d_to_2d(space3d, dist_coef, rot_ang, dist, camera_params, rotation_step=180., rotation_flag=1):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[2]
    cy = camera_params[3]
    space3d = space3d.cpu()

    camera_matrix = torch.tensor([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=torch.float32)
    dist_coeffs = dist_coef

    rotation_angles = []
    if rotation_flag == 1:
        step_deg = np.deg2rad(rotation_step)  # degrees
        for x_angle in np.arange(0, 2 * np.pi, step_deg):
            for y_angle in np.arange(0, 2 * np.pi, step_deg):
                for z_angle in np.arange(0, 2 * np.pi, step_deg):
                    rotation_angles.append((x_angle, y_angle, z_angle))
    else:
        rotation_angles = np.array([(0, 0, 0),
                                    (rot_ang[0], 0, 0),
                                    (0, rot_ang[1], 0),
                                    (0, 0, rot_ang[2]),
                                    (rot_ang[3], rot_ang[4], rot_ang[5]),
                                    (rot_ang[6], rot_ang[7], rot_ang[8]),
                                    (rot_ang[9], rot_ang[10], rot_ang[11]),
                                    (np.pi, np.pi, np.pi)])

    views = []
    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    z_axis = []
    d4 = torch.ones((space3d.shape[0], 1))
    space = space3d
    space3d = torch.cat([space3d, d4], dim=1)
    for angle in rotation_angles:
        # Rotation
        rvec_torch = torch.tensor(angle, dtype=torch.float32)
        # Translation
        tvec_torch = torch.tensor([[-trans_x], [-trans_y], [-trans_z]], dtype=torch.float32)
        # TODO: OpenCV to torch
        # view_2d, _ = cv2.projectPoints(space.detach().numpy(), rvec_torch.detach().numpy(),
        #                                tvec_torch.detach().numpy(),
        #                                camera_matrix.numpy(), dist_coeffs.numpy())
        R = rodrigues(rvec_torch[0], rvec_torch[1], rvec_torch[2], 0)
        RT = torch.cat([R, tvec_torch], dim=1)
        view2d = camera_matrix @ RT @ space3d.T
        # print(view2d[:2, :].T.unsqueeze(1))
        # print(view_2d.shape, viewd2d[:2, :].T.unsqueeze(1).numpy().shape)
        views.append(view2d[:2, :].T.unsqueeze(1).numpy())
        # views.append(viewd2d[:2,:].T.unsqueeze(1).numpy())
        z_axis.append(space3d[:, 2])
    return np.array(views), np.array(z_axis)


def create_2d_views(grid_step, space3d, dist_coef, rot_ang, distances, camera_params, device, rotation_step=180,
                    rotation_flag=0):
    grid_padding = int((1 / grid_step))*2
    img_views = []
    # print(space3d.shape)
    z = []
    for k in range(space3d.shape[0]):
        views, z_axis = project3d_to_2d(space3d[k], dist_coef, rot_ang, distances, camera_params,
                                        rotation_step=rotation_step,
                                        rotation_flag=rotation_flag)

        exploded_views = []
        for j in range(views.shape[0]):
            view = torch.full((int(1 / grid_step) + grid_padding, int(1 / grid_step) + grid_padding), 0.)
            for i in range(views[0].shape[0]):
                idx_x = (views[j][i][0][0] / grid_step).astype(int) + int(grid_padding/2)
                idx_y = (views[j][i][0][1] / grid_step).astype(int) + int(grid_padding/2)
                view[idx_x, idx_y] = 1.
            exploded_views.append(view)
        img_views.append(exploded_views)
        z.append(z_axis)
    return torch.tensor(np.array(img_views), device=device), torch.tensor(np.array(z), device=device)


def project_2d_to_3d(view2d, dist_coef, rot_ang, dist, camera_params, rotation_step=180., rotation_flag=1):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[2]
    cy = camera_params[3]
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32)

    rotation_angles = np.array([(0, 0, 0),
                                (rot_ang[0], 0, 0),
                                (0, rot_ang[1], 0),
                                (0, 0, rot_ang[2]),
                                (rot_ang[3], rot_ang[4], rot_ang[5]),
                                (rot_ang[6], rot_ang[7], rot_ang[8]),
                                (rot_ang[9], rot_ang[10], rot_ang[11]),
                                (np.pi, np.pi, np.pi)])

    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    # _, rvec, tvec = cv2.solvePnP(object_points, view2d.cpu(), camera_matrix.numpy(), dist_coef.numpy())
    views = []
    i = 0
    for angle in rotation_angles:
        # Rotation
        rvec_torch = torch.tensor(angle, dtype=torch.float32)
        # rot_matrix = cv2.Rodrigues(rvec_torch.numpy())

        # Translation
        tvec_torch = torch.tensor([[-trans_x], [-trans_y], [-trans_z]], dtype=torch.float32)
        # Inverse transformation
        # print(view2d[0][i].cpu().numpy().shape, '\n', camera_matrix.numpy().shape, '\n', dist_coef.numpy().shape)
        # plt.imshow(view2d[0][i].cpu())
        # plt.show()
        k = 92
        # print(view2d.shape)
        view2d_tensor = torch.tensor(view2d[0][i])
        # topk_values, topk_indices = torch.topk(view2d_tensor.view(-1), k)
        # y_indices = topk_indices // width
        # x_indices = topk_indices % width
        # pixels = torch.stack((x_indices, y_indices), dim=1).float()
        # print(pixels)

        pixels = torch.nonzero(view2d_tensor > 0.01)

        # undistorted normalized points
        d3_coords_arr = torch.empty((1, 3))
        for j in range(pixels.shape[0]):
            s = 0.01
            grid_padding = int((1 / s))*2

            # xy_undistorted = cv2.undistortPoints(pixels[j].unsqueeze(0).unsqueeze(0).numpy(), camera_matrix.numpy(),
            #                                      dist_coef.numpy())
            # xy = np.array(xy_undistorted)
            # u = xy[0][0][0]
            # v = xy[0][0][1]
            u, v = pixels[j].squeeze()
            K_inv = torch.linalg.inv(K)
            rot_ang_x_deg = rvec_torch[0]
            rot_ang_y_deg = rvec_torch[1]
            rot_ang_z_deg = rvec_torch[2]
            R = rodrigues(rot_ang_x_deg, rot_ang_y_deg, rot_ang_z_deg, 0)

            uv_coords = torch.tensor([[u - grid_padding/2], [v - grid_padding/2], [1.]])

            pc = s * K_inv @ uv_coords
            # T = torch.cat([R_inv, tvec_torch], dim=1)
            pw = tvec_torch + (R @ pc)

            d3_coords = torch.tensor([pw[0][0], pw[1][0], pw[2][0]])
            # print(d3_coords)
            # time.sleep(2)
            d3_coords_arr = torch.cat([d3_coords_arr, d3_coords.unsqueeze(0)], dim=0)
        l = k - d3_coords_arr.shape[0] + 1
        if l > 0:
            perm = torch.randperm(d3_coords_arr.size(0))
            idx = perm[:l]
            sample = d3_coords_arr[idx]
            d3_coords_arr = torch.cat([d3_coords_arr, sample])
        else:
            pass
        views.append(d3_coords_arr[1:])
        i += 1

    views = torch.tensor(np.array(views))

    # fig = plt.figure()
    # prota = fig.add_subplot(111, projection='3d')
    # prota.scatter(views[:, :, 0], views[:, :, 1], views[:, :, 2], c='r')
    # plt.show()
    coords3d = views[0, :, :]
    for m in range(1, views.shape[0]):
        coords3d = torch.cat([coords3d, views[m, :, :]])
    # print(coords3d.shape)
    return coords3d


def rodrigues(rot_ang_x_deg, rot_ang_y_deg, rot_ang_z_deg, inv_flag):
    rot_x, rot_y, rot_z = torch.deg2rad(rot_ang_x_deg), torch.deg2rad(rot_ang_y_deg), torch.deg2rad(rot_ang_z_deg)
    rotation_vector = torch.tensor([rot_x, rot_y, rot_z], dtype=torch.float32)
    theta = torch.norm(rotation_vector)
    if theta.item() == 0:
        return torch.eye(3, dtype=torch.float32)
    rotation_axis = rotation_vector / theta
    K = torch.tensor([[0, -rotation_axis[2], rotation_axis[1]],
                      [rotation_axis[2], 0, -rotation_axis[0]],
                      [-rotation_axis[1], rotation_axis[0], 0]], dtype=torch.float32)

    I = torch.eye(3, dtype=torch.float32)
    rotation_matrix = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.mm(K, K)

    if inv_flag == 1:
        rotation_matrix = torch.inverse(rotation_matrix)
    return rotation_matrix
