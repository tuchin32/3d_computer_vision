import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

if __name__ == '__main__':
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    poses = np.load('pose/q1_pose.npy')

    start_id = 164
    errors = {'rot': [], 't': []}
    for imgid in range(start_id, len(images_df) + 1):
        # Find ground truth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==imgid]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        ### Q1-2. Compute median pose error ###
        # Rotation error
        rot_mat = poses[imgid - start_id, :, :3]
        rot_gt_mat = R.from_quat(rotq_gt).as_matrix()

        rot_e = np.dot(rot_mat, np.linalg.inv(rot_gt_mat.reshape(3, 3)))
        rot_e = R.from_matrix(rot_e).as_quat()
        errors['rot'].append(2 * np.arccos(rot_e[3]))

        # Translation error
        tvec = poses[imgid - start_id, :, 3]
        errors['t'].append(np.linalg.norm(tvec.reshape(1, 3) - tvec_gt))

    err_rot = np.median(errors['rot'])
    err_t = np.median(errors['t'])
    print(f'median pose error, rot = {err_rot}, t = {err_t}')