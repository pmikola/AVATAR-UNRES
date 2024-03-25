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


def project3d_to_2d(space3d, dist_coef, rot_ang, trans, camera_params, device):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[3]
    cy = camera_params[4]
    K = torch.tensor([[fx, 0., cx],
                      [0., fy, cy],
                      [0., 0., 1.]], device=device, dtype=torch.float32)
    dist_coeffs = dist_coef
    # TODO - make nonlinear distortions  and reverse it in 2d to 3d
    # TODO 2 : and add lens equation to camera matrix and reverse it in 2d to 3d
    rotation_angles = torch.tensor([(0, 0, 0),
                                    (rot_ang[0], 0, 0),
                                    (0, rot_ang[1], 0),
                                    (0, 0, rot_ang[2]),
                                    (0, 0, rot_ang[3])], device=device)
    trans_x = trans[0]
    trans_y = trans[1]
    trans_z = trans[2]
    d4 = torch.ones((space3d.shape[0], 1), device=device)
    space3d = torch.cat([space3d, d4], dim=1)
    I = torch.eye(3, device=device, dtype=torch.float32)
    I_full = torch.cat([I, torch.zeros((3, 1), device=device)], dim=1)
    dim_equalizer = torch.tensor([0., 0., 0., 1.], device=device).unsqueeze(0)
    tvec = torch.tensor([[trans_x], [trans_y], [trans_z]], device=device, dtype=torch.float32)
    depth = torch.empty((rotation_angles.shape[0], space3d.shape[0], 1, 1), device=device, dtype=torch.float32)
    views = torch.empty((rotation_angles.shape[0], space3d.shape[0], 1, 2), device=device, dtype=torch.float32)

    for i, rvec in enumerate(rotation_angles):
        R = rodrigues(rvec[0], rvec[1], rvec[2], 0, device)
        RT = torch.cat([R, tvec], dim=1)
        RT = torch.cat([RT, dim_equalizer], dim=0)
        space3d_transformed = (K @ I_full @ RT @ space3d.T)
        lambda_z = space3d_transformed[2, :].unsqueeze(0)
        camera_proj_2d = space3d_transformed[:2, :]  # / lambda_z
        depth[i] = lambda_z.T.unsqueeze(1)
        views[i] = camera_proj_2d.T.unsqueeze(1)
    return views, depth


def create_2d_views(space3d, grid_step, grid_padding, dist_coef, rot_ang, distances, camera_params, device):
    grid = int(1 / grid_step) + int(grid_padding)
    no_quadrants = 4
    no_views = rot_ang.shape[0]
    k = 92
    view2d = torch.empty((space3d.shape[0], no_views, no_quadrants, grid, grid), device=device)
    depth = torch.empty((space3d.shape[0], no_views, k, 1, 1), device=device)
    for k in range(space3d.shape[0]):
        views, z_axis = project3d_to_2d(space3d[k], dist_coef, rot_ang, distances, camera_params, device)
        exploded_views = torch.empty((views.shape[0], no_quadrants, grid, grid), device=device)
        for j in range(no_views):
            # Quadrants
            q_pp = torch.full((grid, grid), 0.0, device=device)
            q_pn = torch.full((grid, grid), 0.0, device=device)
            q_np = torch.full((grid, grid), 0.0, device=device)
            q_nn = torch.full((grid, grid), 0.0, device=device)
            for i in range(views[0].shape[0]):
                idx_x = ((views[j][i][0][0]) / grid_step).int()
                idx_y = ((views[j][i][0][1]) / grid_step).int()
                if abs(idx_x) + int(grid_padding / 2) > grid or abs(idx_y) + int(grid_padding / 2) > grid:
                    pass
                else:
                    if idx_x >= 0 and idx_y >= 0:
                        q_pp[idx_x + int(grid_padding / 2), idx_y + int(grid_padding / 2)] += z_axis[j][i][0][0]
                    elif idx_x > 0 and idx_y < 0:
                        q_pn[idx_x + int(grid_padding / 2), abs(idx_y) + int(grid_padding / 2)] += z_axis[j][i][0][0]
                    elif idx_x < 0 and idx_y > 0:
                        q_np[abs(idx_x) + int(grid_padding / 2), idx_y + int(grid_padding / 2)] += z_axis[j][i][0][0]
                    else:
                        q_nn[abs(idx_x) + int(grid_padding / 2), abs(idx_y) + int(grid_padding / 2)] += z_axis[j][i][0][0]
            exploded_views[j][0] = q_pp
            exploded_views[j][1] = q_pn
            exploded_views[j][2] = q_np
            exploded_views[j][3] = q_nn
        view2d[k] = exploded_views
        depth[k] = z_axis
    return view2d, depth


def project_2d_to_3d(view2d, depth, dist_coef, rot_ang, dist, camera_params, grid_step, grid_padding, device):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[3]
    cy = camera_params[4]
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], device=device, dtype=torch.float32)
    K_inv = torch.linalg.pinv(K)
    dim_equalizer = torch.tensor([0., 0., 0., 1.], device=device).unsqueeze(0)
    I = torch.eye(3, device=device, dtype=torch.float32)
    I_full = torch.cat([I, torch.zeros((3, 1), device=device)], dim=1)
    rotation_angles = torch.tensor([(0, 0, 0),
                                    (rot_ang[0], 0, 0),
                                    (0, rot_ang[1], 0),
                                    (0, 0, rot_ang[2]),
                                    (0, 0, rot_ang[3])], device=device)
    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    k = 92
    no_quadrants = 4
    # _, rvec, tvec = cv2.solvePnP(object_points, view2d.cpu(), camera_matrix.numpy(), dist_coef.numpy())
    c3d = torch.empty((1, 3), device=device)

    # Mirror = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    for i, rvec in enumerate(rotation_angles):

        R = rodrigues(rvec[0], rvec[1], rvec[2], 0, device)
        d3_coords_arr = torch.empty((1, 3), device=device)
        for quadrant in range(view2d[0][i].shape[0]):
            if quadrant == 0:
                h, g = 1., 1.
            elif quadrant == 1:
                h, g = 1., -1.
            elif quadrant == 2:
                h, g = -1., 1.
            else:
                h, g = -1., -1.
            # Translation
            tvec_torch = torch.tensor([[trans_x], [trans_y], [trans_z]], device=device, dtype=torch.float32)
            # Inverse transformation

            view2d_tensor = view2d[0][i][quadrant]
            threshold = 0.002
            pixels_positive = (view2d_tensor > threshold).nonzero()
            pixels_negative = (view2d_tensor < -threshold).nonzero()
            pixels = torch.cat([pixels_positive, pixels_negative], dim=0)
            pixels, pixels_ind = torch.topk(pixels, k=k, dim=0)

            for j in range(pixels.shape[0]):
                s = grid_step
                u, v = pixels[j]
                w = view2d_tensor[u, v]
                points_3d_transformed = torch.tensor(
                    [[h * ((u - grid_padding / 2) * s)], [g * ((v - grid_padding / 2) * s)], [w]], device=device)
                homogenous_coords = torch.cat([points_3d_transformed, torch.ones((1, 1), device=device)], dim=0)

                points_3d_transformed_world = K_inv @ I_full @ homogenous_coords

                points_3d_transformed_world = torch.cat(
                    [points_3d_transformed_world, torch.ones((1, 1), device=device)], dim=0)
                RT_inv = torch.cat([R.T, -R.T @ tvec_torch], dim=1)

                RT_inv = torch.cat([RT_inv, dim_equalizer], dim=0)
                points_3d = RT_inv @ points_3d_transformed_world
                d3_coords = torch.tensor([points_3d[0][0], points_3d[1][0], points_3d[2][0]], device=device)
                d3_coords_arr = torch.cat([d3_coords_arr, d3_coords.unsqueeze(0)], dim=0)

            l = k - d3_coords_arr.shape[0] + 1
            if l > 0:
                perm = torch.randperm(d3_coords_arr.size(0), device=device)
                idx = perm[:l]
                sample = d3_coords_arr[idx]
                d3_coords_arr = torch.cat([d3_coords_arr, sample])
                d3_coords_arr = d3_coords_arr[1:]
            elif d3_coords_arr.shape[0] > k:
                d3_coords_arr = d3_coords_arr[1:]
                indices = torch.randperm(d3_coords_arr.size(0), device=device)[:k]
                d3_coords_arr = d3_coords_arr[indices]
            else:
                pass
        c3d = torch.cat([c3d, d3_coords_arr], dim=0)

    c3d = c3d[1:]
    c3d = c3d[~torch.all(c3d > torch.tensor([1., 1., 1.], device=device), dim=1)]
    c3d = c3d[~torch.all(c3d < torch.tensor([0., 0., 0.], device=device), dim=1)]
    c3d_out = torch.empty((1, 3), device=device)
    sub_diff = torch.abs(torch.subtract(c3d.unsqueeze(1), c3d.unsqueeze(0)))
    threshold = 0.005

    matches = sub_diff < threshold
    all_true_mask = torch.all(matches, dim=2)
    idxt = torch.nonzero(all_true_mask.view(all_true_mask.shape[0], -1), as_tuple=False)
    unique_rows, counts = idxt[:, 0].unique(return_counts=True)
    filtered_idx = unique_rows[counts > 4]
    for idx in filtered_idx:
        matches_idx = idxt[idxt[:, 0] == idx][:, 1]
        c3d_out = torch.cat([c3d_out, torch.mean(c3d[matches_idx], dim=0, keepdim=True)], dim=0)
    if c3d_out.shape[0] < 1:
        pass
    else:
        c3d_out = c3d_out[1:]
    if c3d_out.shape[0] < k:
        l = k - c3d_out.shape[0]
        indices = torch.randint(high=c3d_out.shape[0], size=(l,))
        c3d_out = torch.cat([c3d_out, c3d_out[indices]])
    elif c3d_out.shape[0] > k:
        # unique_tenor = torch.unique(c3d_out, dim=1)
        # print(unique_tenor.shape[0] == c3d_out.shape[0])
        c3d_out = c3d_out[:k]
    else:
        pass

    return c3d_out


def rodrigues(rot_x, rot_y, rot_z, inv_flag, device):
    # rotation_vector = torch.tensor([rot_x, rot_y, rot_z],
    #                                dtype=torch.float32)
    # theta = torch.norm(rotation_vector)
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

    if not inv_flag:
        sin_rx, cos_rx = torch.sin(rot_x), torch.cos(rot_x)
        sin_ry, cos_ry = torch.sin(rot_y), torch.cos(rot_y)
        sin_rz, cos_rz = torch.sin(rot_z), torch.cos(rot_z)
        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]], device=device)
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]], device=device)
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]], device=device)
        rotation_matrix = R_x @ R_y @ R_z
    else:
        # rotation_matrix = torch.linalg.pinv(rotation_matrix)
        sin_rx, cos_rx = torch.sin(rot_x), torch.cos(rot_x)
        sin_ry, cos_ry = torch.sin(rot_y), torch.cos(rot_y)
        sin_rz, cos_rz = torch.sin(rot_z), torch.cos(rot_z)

        R_x = torch.tensor([[1, 0, 0],
                            [0, cos_rx, -sin_rx],
                            [0, sin_rx, cos_rx]], device=device)
        R_y = torch.tensor([[cos_ry, 0, sin_ry],
                            [0, 1, 0],
                            [-sin_ry, 0, cos_ry]], device=device)
        R_z = torch.tensor([[cos_rz, -sin_rz, 0],
                            [sin_rz, cos_rz, 0],
                            [0, 0, 1]], device=device)
        rotation_matrix = R_x @ R_y @ R_z
        rotation_matrix = rotation_matrix.T
    return rotation_matrix
