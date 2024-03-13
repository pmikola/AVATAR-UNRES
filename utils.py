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
    cx = camera_params[3]
    cy = camera_params[4]
    space3d = space3d.cpu()
    K = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], dtype=torch.float32)
    dist_coeffs = dist_coef
    rotation_angles = np.array([(0, 0, 0),
                                (rot_ang[0], 0, 0),
                                (rot_ang[1], 0, 0),
                                (rot_ang[2], 0, 0)])

    views = []
    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    depth = []

    d4 = torch.ones((space3d.shape[0], 1))
    space3d = torch.cat([space3d, d4], dim=1)
    for angle in rotation_angles:
        rvec_torch = torch.tensor(angle, dtype=torch.float32)
        tvec_torch = torch.tensor([[trans_x], [trans_y], [trans_z]], dtype=torch.float32)
        R = rodrigues(rvec_torch[0], rvec_torch[1], rvec_torch[2], 0)
        R_full = torch.cat([R, torch.zeros((1, 3))], dim=0)
        T_full = torch.cat([tvec_torch, torch.ones((1, 1))], dim=0)
        I = torch.eye(3, dtype=torch.float32)
        I_full = torch.cat([I, torch.zeros((3, 1))], dim=1)
        RT = torch.cat([R_full, T_full], dim=1)
        # space3d_transformed = K @ RT @ space3d.T  # + tvec_torch)
        P = K @ I_full @ RT
        space3d_transformed = P @ space3d.T  # + tvec_torch)
        lambda_z = space3d_transformed[2, :].unsqueeze(0)
        depth.append(lambda_z.T.unsqueeze(1).numpy())
        camera_proj_2d = space3d_transformed[:2, :]  # / lambda_z
        # print(view_2d.shape, viewd2d[:2, :].T.unsqueeze(1).numpy().shape)
        views.append(camera_proj_2d.T.unsqueeze(1).numpy())
        # views.append(viewd2d[:2,:].T.unsqueeze(1).numpy())qwq
    return np.array(views), np.array(depth)


def create_2d_views(space3d,grid_step, grid_padding,  dist_coef, rot_ang, distances, camera_params, device):
    img_views = []
    # print(space3d.shape)
    z_scale = []
    for k in range(space3d.shape[0]):
        views, z_axis = project3d_to_2d(space3d[k], dist_coef, rot_ang, distances, camera_params)
        exploded_views = []

        for j in range(views.shape[0]):
            view = torch.full((int(1 / grid_step) + int(grid_padding), int(1 / grid_step) + int(grid_padding)), 0.)

            for i in range(views[0].shape[0]):
                idx_x = (views[j][i][0][0] / grid_step).astype(int) + int(grid_padding / 2)
                idx_y = (views[j][i][0][1] / grid_step).astype(int) + int(grid_padding / 2)
                # z_axis[j][i][0][0] = (z_axis[j][i][0][0] / grid_step).astype(int) + int(grid_padding / 2)
                view[idx_x, idx_y] += 1.
            exploded_views.append(view)
        img_views.append(exploded_views)
        z_scale.append(z_axis)
    return torch.tensor(np.array(img_views), device=device), torch.tensor(np.array(z_scale), device=device)


def project_2d_to_3d(view2d, depth, dist_coef, rot_ang, dist, camera_params, grid_step, grid_padding,
                     rotation_step=180.,
                     rotation_flag=1):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[2]
    cy = camera_params[3]
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=torch.float32)
    K_inv = torch.linalg.pinv(K)

    rotation_angles = np.array([(0, 0, 0),
                                (rot_ang[0], 0, 0),
                                (rot_ang[1], 0, 0),
                                (rot_ang[2], 0, 0)])
    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    # _, rvec, tvec = cv2.solvePnP(object_points, view2d.cpu(), camera_matrix.numpy(), dist_coef.numpy())
    views = []
    i = 0
    Mirror = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

    for angle in rotation_angles:
        # Rotation
        rvec_torch = torch.tensor(angle, dtype=torch.float32)
        # rot_matrix = cv2.Rodrigues(rvec_torch.numpy())

        # Translation
        tvec_torch = torch.tensor([[trans_x], [trans_y], [trans_z]], dtype=torch.float32)
        # Inverse transformation
        # print(view2d[0][i].cpu().numpy().shape, '\n', camera_matrix.numpy().shape, '\n', dist_coef.numpy().shape)
        # plt.imshow(view2d[0][i].cpu())
        # plt.show()
        k = 92
        view2d_tensor = view2d[0][i]
        z_tensor = depth[0][i].squeeze(2)
        width = view2d_tensor.shape[1]
        print(view2d_tensor.shape) # TODO grid_padding/2 range inside view2d
        topk_values, topk_indices = torch.topk(view2d_tensor.view(-1), k)
        # unravel
        x_indices = topk_indices / width
        y_indices = topk_indices % width
        pixels = torch.stack((x_indices, y_indices), dim=1).float()
        # pixels = torch.nonzero(view2d_tensor > 0.1)
        # undistorted normalized points

        d3_coords_arr = torch.empty((1, 3))
        for j in range(pixels.shape[0]):
            s = grid_step
            # xy_undistorted = cv2.undistortPoints(pixels[j].unsqueeze(0).unsqueeze(0).numpy(), camera_matrix.numpy(),
            #                                      dist_coef.numpy())
            # xy = np.array(xy_undistorted)
            # u = xy[0][0][0]
            # v = xy[0][0][1]

            u, v = pixels[j]

            w = z_tensor[j]
            # w = int(w/s + grid_padding / 2)
            homogenous_coords = torch.tensor(
                [[(u - grid_padding/2) * s], [(v - grid_padding/2) * s], [w], [1.]])
            rot_ang_x_deg = rvec_torch[0]
            rot_ang_y_deg = rvec_torch[1]
            rot_ang_z_deg = rvec_torch[2]
            R = rodrigues(rot_ang_x_deg, rot_ang_y_deg, rot_ang_z_deg, True)

            t_inv = -R @ tvec_torch
            R_full = torch.cat([R, torch.zeros((1, 3))], dim=0)
            T_full = torch.cat([t_inv, torch.ones((1, 1))], dim=0)
            I = torch.eye(3, dtype=torch.float32)
            I_full = torch.cat([I, torch.zeros((3, 1))], dim=1)
            RT_inv = torch.cat([R_full, T_full], dim=1)

            world_coords = K_inv @ I_full @ Mirror @ RT_inv @ homogenous_coords
            # world_coords = R @ K_inv @ homogenous_coords - tvec_torch

            # world_coords = K_inv @ (homogenous_coords)
            # normalized_with_depth = normalized_coords

            # world_coords = normalized_with_depth

            # pc = s * K_inv @ uv_coords
            # pw = tvec_torch + (R @ pc)
            # pw = R @ (s * K_inv @ uv_coords - tvec_torch)
            # world_coords = R@(K_inv@homogenous_coords - tvec_torch)

            d3_coords = torch.tensor([world_coords[0][0], world_coords[1][0], world_coords[2][0]])

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

    # coords3d = views[0, :, :]
    # for m in range(1, views.shape[0]):
    #     coords3d = torch.cat([coords3d, views[m, :, :]])
    return views


def rodrigues(rot_x, rot_y, rot_z, inv_flag):

    rotation_vector = torch.tensor([rot_x, rot_y, rot_z], dtype=torch.float32)
    theta = torch.norm(rotation_vector)
    # if theta.item() == 0:
    #     return torch.eye(3, dtype=torch.float32)
    # #
    # rotation_axis = rotation_vector / theta
    # R = torch.tensor([[0., -rotation_axis[2], rotation_axis[1]],
    #                   [rotation_axis[2], 0, -rotation_axis[0]],
    #                   [-rotation_axis[0], rotation_axis[1], 0.]], dtype=torch.float32)
    # #
    # I = torch.eye(3, dtype=torch.float32)
    # rotation_matrix = I + torch.sin(theta) * R + (1 - torch.cos(theta)) * R @ R
    #


    if not inv_flag:
        sin_rx, cos_rx = np.sin(rot_x), np.cos(rot_x)
        sin_ry, cos_ry = np.sin(rot_y), np.cos(rot_y)
        sin_rz, cos_rz = np.sin(rot_z), np.cos(rot_z)
        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]])
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]])
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]])
        rotation_matrix = R_x @ R_y @ R_z
    else:
        # rotation_matrix = torch.linalg.pinv(rotation_matrix)
        sin_rx, cos_rx = np.sin(rot_x), np.cos(rot_x)
        sin_ry, cos_ry = np.sin(rot_y), np.cos(rot_y)
        sin_rz, cos_rz = np.sin(rot_z), np.cos(rot_z)

        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]])
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]])
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]])
        rotation_matrix = R_z @ R_y @ R_x
    return rotation_matrix
