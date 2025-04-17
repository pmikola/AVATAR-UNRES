import os
import time

import cv2
import numpy as np
import torch
from Bio.PDB import PPBuilder, PDBParser
from Bio.PDB.Atom import Atom
from matplotlib import pyplot as plt
from torch.nn.functional import relu
from torch.multiprocessing import Pool


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
                      [0., 0., 1.]], device=device, dtype=torch.float32, requires_grad=True)
    dist_coeffs = dist_coef
    # TODO - make nonlinear distortions  and reverse it in 2d to 3d
    # TODO 2 : and add lens equation to camera matrix and reverse it in 2d to 3d
    rotation_angles = torch.tensor([(0, 0, 0),
                                    (rot_ang[0], 0, 0),
                                    (0, rot_ang[1], 0),
                                    (0, 0, rot_ang[2]),
                                    (0, 0, rot_ang[3])], device=device, requires_grad=True)
    trans_x = trans[0]
    trans_y = trans[1]
    trans_z = trans[2]
    d4 = torch.ones((space3d.shape[0], 1), device=device, requires_grad=True)
    space3d = torch.cat([space3d, d4], dim=1)
    I = torch.eye(3, device=device, dtype=torch.float32, requires_grad=True)
    I_full = torch.cat([I, torch.zeros((3, 1), device=device, requires_grad=True)], dim=1)
    dim_equalizer = torch.tensor([0., 0., 0., 1.], device=device, requires_grad=True).unsqueeze(0)
    tvec = torch.tensor([[trans_x], [trans_y], [trans_z]], device=device, dtype=torch.float32, requires_grad=True)
    depth = torch.empty((rotation_angles.shape[0], space3d.shape[0], 1, 1), device=device, dtype=torch.float32,
                        requires_grad=True)
    views = torch.empty((rotation_angles.shape[0], space3d.shape[0], 1, 2), device=device, dtype=torch.float32,
                        requires_grad=True)

    R_list = [rodrigues(rvec[0], rvec[1], rvec[2], 0, device) for rvec in rotation_angles]
    R_stacked = torch.stack(R_list)
    tvec_stacked = tvec.unsqueeze(0).repeat(R_stacked.shape[0], 1, 1)
    dim_equalizer_stacked = dim_equalizer.unsqueeze(0).repeat(R_stacked.shape[0], 1, 1)
    RT = torch.cat([R_stacked, tvec_stacked], dim=2)
    RT = torch.cat([RT, dim_equalizer_stacked], dim=1)
    space3d_transformed = (K @ I_full @ RT @ space3d.T)
    depth.data = space3d_transformed[:, 2, :].unsqueeze(2).unsqueeze(3)  # ([5, 92, 1, 1])
    views.data = space3d_transformed[:, :2, :].unsqueeze(2).permute(0, 3, 2, 1)  # / lambda_z ([5, 92, 1, 2])
    return views, depth


def create_2d_views(space3d, grid_step, grid_padding, dist_coef, rot_ang, distances, camera_params, device):
    grid = int(1 / grid_step) + int(grid_padding)
    no_quadrants = 4
    no_views = rot_ang.shape[0]
    k = 92
    view2d = torch.empty((space3d.shape[0], no_views, no_quadrants, grid, grid), device=device, requires_grad=True)
    depth = torch.empty((space3d.shape[0], no_views, k, 1, 1), device=device, requires_grad=True)
    for k in range(space3d.shape[0]):
        views, z_axis = project3d_to_2d(space3d[k], dist_coef, rot_ang, distances, camera_params, device)
        exploded_views = torch.empty((views.shape[0], no_quadrants, grid, grid), device=device, requires_grad=True)
        for j in range(no_views):
            # Quadrants
            q_pp = torch.full((grid, grid), 0.0, device=device, requires_grad=True)
            q_pn = torch.full((grid, grid), 0.0, device=device, requires_grad=True)
            q_np = torch.full((grid, grid), 0.0, device=device, requires_grad=True)
            q_nn = torch.full((grid, grid), 0.0, device=device, requires_grad=True)
            for i in range(views[0].shape[0]):
                idx_x = ((views[j][i][0][0]) / grid_step).int()
                idx_y = ((views[j][i][0][1]) / grid_step).int()
                if abs(idx_x) + int(grid_padding / 2) > grid or abs(idx_y) + int(grid_padding / 2) > grid:
                    pass
                else:
                    if idx_x >= 0 and idx_y >= 0:
                        q_pp.data[idx_x + int(grid_padding / 2), idx_y + int(grid_padding / 2)] = q_pp[idx_x + int(
                            grid_padding / 2), idx_y + int(grid_padding / 2)] + z_axis[j][i][0][0]
                    elif idx_x > 0 and idx_y < 0:
                        q_pn.data[idx_x + int(grid_padding / 2), abs(idx_y) + int(grid_padding / 2)] = q_pn[idx_x + int(
                            grid_padding / 2), abs(idx_y) + int(grid_padding / 2)] + z_axis[j][i][0][0]
                    elif idx_x < 0 and idx_y > 0:
                        q_np.data[abs(idx_x) + int(grid_padding / 2), idx_y + int(grid_padding / 2)] = q_np[
                                                                                                           abs(idx_x) + int(
                                                                                                               grid_padding / 2), idx_y + int(
                                                                                                               grid_padding / 2)] + \
                                                                                                       z_axis[j][i][0][
                                                                                                           0]
                    else:
                        q_nn.data[abs(idx_x) + int(grid_padding / 2), abs(idx_y) + int(grid_padding / 2)] = q_nn[
                                                                                                                abs(idx_x) + int(
                                                                                                                    grid_padding / 2), abs(
                                                                                                                    idx_y) + int(
                                                                                                                    grid_padding / 2)] + \
                                                                                                            z_axis[j][
                                                                                                                i][0][0]
            exploded_views.data[j][0] = q_pp
            exploded_views.data[j][1] = q_pn
            exploded_views.data[j][2] = q_np
            exploded_views.data[j][3] = q_nn
        view2d.data[k] = exploded_views
        depth.data[k] = z_axis
    return view2d, depth


def project_2d_to_3d(view2d, depth, dist_coef, rot_ang, dist, camera_params, grid_step, grid_padding, device):
    fx = camera_params[0]
    fy = camera_params[1]
    cx = camera_params[3]
    cy = camera_params[4]
    K = torch.tensor([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], device=device, dtype=torch.float32, requires_grad=True)
    K_inv = torch.linalg.pinv(K)
    dim_equalizer = torch.tensor([0., 0., 0., 1.], device=device, requires_grad=True).unsqueeze(0)
    I = torch.eye(3, device=device, dtype=torch.float32, requires_grad=True)
    I_full = torch.cat([I, torch.zeros((3, 1), device=device, requires_grad=True)], dim=1)
    rotation_angles = torch.tensor([(0, 0, 0),
                                    (rot_ang[0], 0, 0),
                                    (0, rot_ang[1], 0),
                                    (0, 0, rot_ang[2]),
                                    (0, 0, rot_ang[3])], device=device, requires_grad=True)
    trans_x = dist[0]
    trans_y = dist[1]
    trans_z = dist[2]
    k = 92
    # _, rvec, tvec = cv2.solvePnP(object_points, view2d.cpu(), camera_matrix.numpy(), dist_coef.numpy())
    out = torch.empty((1, k, 3), device=device, requires_grad=True)
    c3d = torch.empty((1, 3), device=device, requires_grad=True)
    # Mirror = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    for nbatch in range(view2d.shape[0]):
        for i, rvec in enumerate(rotation_angles):
            R = rodrigues(rvec[0], rvec[1], rvec[2], 0, device)
            d3_coords_arr = torch.empty((1, 3), device=device, requires_grad=True)
            for quadrant in range(view2d[nbatch][i].shape[0]):
                if quadrant == 0:
                    h, g = 1., 1.
                elif quadrant == 1:
                    h, g = 1., -1.
                elif quadrant == 2:
                    h, g = -1., 1.
                else:
                    h, g = -1., -1.
                # Translation
                tvec_torch = torch.tensor([[trans_x], [trans_y], [trans_z]], device=device, dtype=torch.float32,
                                          requires_grad=True)
                # Inverse transformation

                view2d_tensor = view2d[nbatch][i][quadrant]
                threshold = 0.05 # parametr to learn in the future
                pixels_positive = (view2d_tensor > threshold).nonzero()
                pixels_negative = (view2d_tensor < -threshold).nonzero()
                pixels = torch.cat([pixels_positive, pixels_negative], dim=0)
                kk = k

                # if pixels.shape[0] < k:
                #     kk = pixels.shape[0]
                # pixels, pixels_ind = torch.topk(pixels, k=kk, dim=0)
                # TODO need to change this to something that will work with test loop without model and with model
                for j in range(pixels.shape[0]):
                    s = grid_step
                    u, v = pixels[j]
                    w = view2d_tensor[u, v]
                    points_3d_transformed = torch.tensor(
                        [[h * ((u - grid_padding / 2) * s)], [g * ((v - grid_padding / 2) * s)], [w]], device=device,
                        requires_grad=True)
                    homogenous_coords = torch.cat(
                        [points_3d_transformed, torch.ones((1, 1), requires_grad=True, device=device)], dim=0)
                    points_3d_transformed_world = K_inv @ I_full @ homogenous_coords
                    points_3d_transformed_world = torch.cat(
                        [points_3d_transformed_world, torch.ones((1, 1), requires_grad=True, device=device)], dim=0)
                    RT_inv = torch.cat([R.T, -R.T @ tvec_torch], dim=1)
                    RT_inv = torch.cat([RT_inv, dim_equalizer], dim=0)
                    points_3d = RT_inv @ points_3d_transformed_world
                    d3_coords = torch.tensor([points_3d[0][0], points_3d[1][0], points_3d[2][0]], device=device)
                    d3_coords_arr = torch.cat([d3_coords_arr, d3_coords.unsqueeze(0)], dim=0)

                if d3_coords_arr.shape[0] < k:
                    l = k - d3_coords_arr.shape[0]
                    indices = torch.randint(high=d3_coords_arr.shape[0], size=(l,))
                    d3_coords_arr = torch.cat([d3_coords_arr, d3_coords_arr[indices]])
                elif d3_coords_arr.shape[0] > k:
                    d3_coords_arr = d3_coords_arr[1:]
                    indices = torch.randperm(d3_coords_arr.shape[0], device=device)[:k]
                    d3_coords_arr = d3_coords_arr[indices]
                else:
                    pass
            c3d = torch.cat([c3d, d3_coords_arr], dim=0)
        c3d = c3d[1:]
        c3d = c3d[~torch.all(c3d > torch.tensor([1., 1., 1.], device=device), dim=1)]
        c3d = c3d[~torch.all(c3d < torch.tensor([0., 0., 0.], device=device), dim=1)]
        c3d_out = torch.empty((1, 3), device=device)
        sub_diff = torch.abs(torch.subtract(c3d.unsqueeze(1), c3d.unsqueeze(0)))
        threshold = 0.05  # parametr to learn in the future
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
        out = torch.cat([out, c3d_out.unsqueeze(0)], dim=0)
    return out[1:].squeeze(0)


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

    if not inv_flag:
        pass
    else:
        rotation_matrix = rotation_matrix.T
    return rotation_matrix


import os
import numpy as np
from Bio.PDB import PDBParser, PPBuilder

def clone_atom_as_cb(ca_atom):
    from Bio.PDB.Atom import Atom
    original_residue = ca_atom.get_parent()
    new_atom = Atom(
        name="CB",
        coord=ca_atom.coord,
        bfactor=ca_atom.bfactor,
        occupancy=ca_atom.occupancy,
        altloc=ca_atom.altloc,
        fullname=" CB ",
        serial_number=ca_atom.serial_number,
        element="C"
    )
    if original_residue is not None:
        original_residue.add(new_atom)

    return new_atom

def load_unres_ca_sc_atoms(
    pdb_path,
    total_residues=46,
    skip_first_dummy=False,
    skip_last_dummy=False
):

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("REF", pdb_path)

    model = next(structure.get_models())
    chain = next(model.get_chains())
    all_residues = list(chain.get_residues())

    start_idx = 1 if skip_first_dummy else 0
    end_idx   = -1 if skip_last_dummy else len(all_residues)
    real_res  = all_residues[start_idx:end_idx]

    if len(real_res) != total_residues:
        print(f"WARNING: We found {len(real_res)} real residues, expected {total_residues}.")

    ref_atoms = []

    for residue in real_res:
        resname = residue.resname.strip().upper()
        ca_atom = None
        cb_atom = None

        for atom in residue:
            if atom.is_disordered():
                atom = atom.disordered_select(atom.disordered_get_id_list()[0])

            aname = atom.get_name().upper()
            if aname == "CA":
                ca_atom = atom
            elif aname == "CB":
                cb_atom = atom

        if ca_atom is None:
            print(f"Residue {resname} missing CA. Skipping.")
            continue

        ref_atoms.append(ca_atom)

        if cb_atom is None:
            print(f"{resname} missing CB. Duplicating CA for side chain.")
            cb_atom = clone_atom_as_cb(ca_atom)
        ref_atoms.append(cb_atom)

    return ref_atoms

def write_multi_model_pdb(positions, ref_atoms, out_pdb="trajectory.pdb"):
    num_frames, num_atoms, _ = positions.shape
    if num_atoms != len(ref_atoms):
        raise ValueError(f"Mismatch: positions has {num_atoms} atoms, "
                         f"but ref_atoms has {len(ref_atoms)}.")

    with open(out_pdb, "w") as fh:
        for frame_idx in range(num_frames):
            fh.write(f"MODEL     {frame_idx+1}\n")

            for atom_idx in range(num_atoms):
                x, y, z = positions[frame_idx, atom_idx]
                atom = ref_atoms[atom_idx]
                residue = atom.get_parent()
                chain   = residue.get_parent()

                atom_name = atom.get_name()
                res_name  = residue.resname.strip()
                chain_id  = chain.id
                res_seq   = residue.id[1]
                icode     = residue.id[2].strip() if residue.id[2] else ''
                element   = atom.element.strip()
                occupancy = atom.get_occupancy() or 1.0
                bfactor   = atom.get_bfactor()   or 0.0
                serial    = atom_idx + 1  # simple approach


                line = (
                    f"ATOM  {serial:5d} {atom_name:>4s} {res_name:3s} {chain_id:1s}"
                    f"{res_seq:4d}{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{bfactor:6.2f}"
                    f"          {element:>2s}\n"
                )
                fh.write(line)

            fh.write("ENDMDL\n")
        fh.write("END\n")


def write_single_frame_pdbs(positions, ref_atoms, out_dir="frames", prefix="frame"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    num_frames, num_atoms, _ = positions.shape
    if num_atoms != len(ref_atoms):
        raise ValueError("Mismatch between positions and reference atoms length.")

    for frame_idx in range(num_frames):
        pdb_path = os.path.join(out_dir, f"{prefix}_{frame_idx+1:04d}.pdb")
        with open(pdb_path, "w") as fh:
            fh.write(f"MODEL     {frame_idx+1}\n")

            for atom_idx in range(num_atoms):
                x, y, z = positions[frame_idx, atom_idx]
                atom = ref_atoms[atom_idx]
                residue = atom.get_parent()
                chain   = residue.get_parent()

                # metadata
                atom_name = atom.get_name()
                res_name  = residue.resname.strip()
                chain_id  = chain.id
                res_seq   = residue.id[1]
                icode     = residue.id[2].strip() if residue.id[2] else ''
                element   = atom.element.strip()
                occupancy = atom.get_occupancy() or 1.0
                bfactor   = atom.get_bfactor()   or 0.0
                serial    = atom_idx + 1

                line = (
                    f"ATOM  {serial:5d} {atom_name:>4s} {res_name:3s} {chain_id:1s}"
                    f"{res_seq:4d}{icode:1s}   "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"{occupancy:6.2f}{bfactor:6.2f}"
                    f"          {element:>2s}\n"
                )
                fh.write(line)
            fh.write("ENDMDL\n")
            fh.write("END\n")

def process_pdb_and_generate_animations(
    pdb_path,
    pred_positions,
    gt_positions,
    output_folder="unres_output_pdbs",
    separate_frames=False
):
    if not os.path.isfile(pdb_path):
        raise FileNotFoundError(f"Reference PDB not found: {pdb_path}")

    ref_atoms = load_unres_ca_sc_atoms(pdb_path, total_residues=46)

    if len(ref_atoms) != 92:
        print(f"WARNING: your reference list has {len(ref_atoms)} atoms, not 92. Check data consistency.")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not separate_frames:
        gt_pdb_path   = os.path.join(output_folder, "ground_truth_trajectory.pdb")
        pred_pdb_path = os.path.join(output_folder, "predicted_trajectory.pdb")

        print(f"Writing multi-model PDB for GT => {gt_pdb_path}")
        write_multi_model_pdb(gt_positions, ref_atoms, out_pdb=gt_pdb_path)

        print(f"Writing multi-model PDB for Pred => {pred_pdb_path}")
        write_multi_model_pdb(pred_positions, ref_atoms, out_pdb=pred_pdb_path)

    else:
        gt_dir   = os.path.join(output_folder, "ground_truth_frames")
        pred_dir = os.path.join(output_folder, "predicted_frames")

        print(f"Writing single-frame PDBs for GT => {gt_dir}")
        write_single_frame_pdbs(gt_positions, ref_atoms, out_dir=gt_dir, prefix="gt_frame")

        print(f"Writing single-frame PDBs for Pred => {pred_dir}")
        write_single_frame_pdbs(pred_positions, ref_atoms, out_dir=pred_dir, prefix="pred_frame")

    print("Done. Check your output in:", output_folder)


def compute_rmsd(coordsA, coordsB, align=False):
    if coordsA.shape != coordsB.shape:
        raise ValueError("Shape mismatch between coordsA and coordsB")

    if not align:
        diff = coordsA - coordsB
        return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    else:
        A_center = coordsA.mean(axis=0)
        B_center = coordsB.mean(axis=0)
        A = coordsA - A_center
        B = coordsB - B_center

        H = B.T @ A
        U, S, Vt = np.linalg.svd(H)
        R = (U @ Vt).T

        A_aligned = A @ R

        diff = A_aligned - B
        return np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))


import numpy as np
from Bio.PDB import PDBParser


def load_multimodel_pdb_coords(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("traj", pdb_path)

    all_coords = []
    for model in structure:
        model_coords = []
        for chain in model:
            for residue in chain:
                if residue.id[0].strip():
                    continue
                for atom in residue:
                    if atom.is_disordered():
                        atom = atom.disordered_select(atom.disordered_get_id_list()[0])
                    model_coords.append(atom.coord)

        model_coords = np.array(model_coords, dtype=np.float32)
        all_coords.append(model_coords)

    coords = np.stack(all_coords, axis=0)
    return coords


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

