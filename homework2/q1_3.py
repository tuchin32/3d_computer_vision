import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B
    return axes

def load_pyramid(pose):
    rot_mat = pose[:, :3]
    tvec = pose[:, 3]

    axes_c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes_w = np.dot(np.linalg.inv(rot_mat), axes_c.T)
    
    xw = 0.2 * axes_w[:, 0] / np.linalg.norm(axes_w[:, 0])
    yw = 0.2 * axes_w[:, 1] / np.linalg.norm(axes_w[:, 1])
    zw = 0.2 * axes_w[:, 2] / np.linalg.norm(axes_w[:, 2])

    center = -np.dot(np.linalg.inv(rot_mat), tvec)
    ctr_proj = center + zw

    base_w0 = ctr_proj + xw + yw
    base_w1 = ctr_proj + xw - yw
    base_w2 = ctr_proj - xw + yw
    base_w3 = ctr_proj - xw - yw

    points = [center, base_w0, base_w1, base_w2, base_w3]
    lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
    colors = [[0.44, 0.26, 0.08] for i in range(len(lines))]

    pyramid = o3d.geometry.LineSet()
    pyramid.points = o3d.utility.Vector3dVector(points)
    pyramid.lines  = o3d.utility.Vector2iVector(lines)
    pyramid.colors = o3d.utility.Vector3dVector(colors)

    return pyramid, center

def load_trajectory(camera_ctr, num_img=130, start_id=164):
    # Train
    # id = [83, 158, 6, 17, 28, 39, 50, 61, 72, 84, 95, 106, 117, 128, 139, 150,
    #       155, 156, 157, 159, 160, 161, 162, 163]
    # id_sort = [0] + sorted(id)
    # id_list = id
    # for i in range(len(id_sort) - 1):
    #     id_list += [j for j in range(id_sort[i] + 1, id_sort[i + 1])]

    # Valid
    id = [252, 164, 175, 186, 197, 208, 219, 230, 241,
          253, 264, 275, 286, 288, 289, 290, 291, 292, 293]
    id_list = id

    for j in range(1, 13):
        if j == 8:
            id_list += [i for i in range(id[j] + 1, id[j + 1] - 1)]
        else:
            id_list += [i for i in range(id[j] + 1, id[j + 1])]
    
    lines = [[id_list[i] - start_id, id_list[i + 1] - start_id] for i in range(num_img - 1)]
    colors = [[1, 0.84, 0] for i in range(len(lines))]

    traj = o3d.geometry.LineSet()
    traj.points = o3d.utility.Vector3dVector(camera_ctr)
    traj.lines  = o3d.utility.Vector2iVector(lines)
    traj.colors = o3d.utility.Vector3dVector(colors)
    return traj

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)
    return transform_mat

if __name__ == '__main__':
    # load pose and reprojection points
    poses = np.load('pose/q1_pose.npy')

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)

    # load camera pose
    num_image = 130
    camera_ctr = []
    for i in range(num_image):
        pyramid, center = load_pyramid(poses[i])
        vis.add_geometry(pyramid)
        camera_ctr.append(center)

    # load camera trajectory
    trajectory = load_trajectory(camera_ctr)
    vis.add_geometry(trajectory)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    vis.run()
    vis.destroy_window()