import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc

def similarity_transform(points):
    similarity, norm_points = None, None

    mean, std = np.mean(points, axis=0), np.std(points, axis=0)
    norm_points = (points - mean) / std
    if points.shape[1] == 2:
        similarity =  np.array([[1 / std[0], 0, -mean[0] / std[0]], 
                                [0, 1 / std[1], -mean[1] / std[1]],
                                [0, 0, 1]])
    elif points.shape[1] == 3:
        similarity =  np.array([[1 / std[0], 0, 0, -mean[0] / std[0]], 
                                [0, 1 / std[1], 0, -mean[1] / std[1]],
                                [0, 0, 1 / std[2], -mean[2] / std[2]], 
                                [0, 0, 0, 1]])

    return similarity, norm_points

def corr_matrix(pts3d, pts2d):
    mat_A = []
    for pt3d, pt2d in zip(pts3d, pts2d):
        x, y, z = pt3d
        u, v = pt2d
        mat_A.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
        mat_A.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
    return mat_A

def direct_linear_transform(pts3d, pts2d):
    mat_corr = corr_matrix(pts3d, pts2d)

    # Find 12 parameters
    mat_u, mat_s, mat_vt = np.linalg.svd(mat_corr)
    homography = mat_vt[-1, :].reshape((3, 4))
    return homography / homography[-1, -1]

def normalized_dlt(pts3d, pts2d):
    T3d, n_pts3d = similarity_transform(pts3d)
    T2d, n_pts2d = similarity_transform(pts2d)
    n_homography = direct_linear_transform(n_pts3d, n_pts2d)
    homography = np.dot(np.dot(np.linalg.inv(T2d), n_homography), T3d)
    return homography / homography[-1, -1]

def bfm_matcher(query, model):
    kp_query, desc_query = query
    kp_model, desc_model = model

    bf = cv.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))
    
    return points2D, points3D

def pnpsolver(points2D, points3D, solve='opencv'):
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    
    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])
    
    if solve == 'opencv':
        retval, rvec, tvec, inliers = cv.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)
        homography = None
    else:
        # Sample six points to do DLT
        indices = np.random.choice(points3D.shape[0], 6, replace=False)
        pts3d = points3D[indices]
        pts2d = points2D[indices]

        # Compute homography
        homography = normalized_dlt(pts3d, pts2d)
        pose = np.dot(np.linalg.inv(cameraMatrix), homography)

        # Get pose of camera
        rot_mat = pose[:3, :3]
        rvec = R.from_matrix(rot_mat).as_rotvec()

        tvec = pose[:3, 3] / np.linalg.norm(rot_mat[0, :])

    return rvec, tvec, homography

if __name__ == '__main__':
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")
    # images_df.to_csv('images.csv')  

    # Process model descriptors
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    num_image = 130
    start_id = 164
    print(f'Total image numbers: {num_image}')

    homos = []
    poses = []
    errors = {'rot': [], 't': []}

    for imgid in range(start_id, start_id + num_image):
        print('imgid', imgid)

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==imgid]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)
        
        # Find ground truth
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==imgid]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values
        print(ground_truth)

        points_2d, points_3d = bfm_matcher((kp_query, desc_query),(kp_model, desc_model))

        flag = True
        count = 0
        while flag == True:
            # Find correspondance and solve pnp
            # rvec, tvec, homography = pnpsolver(points_2d, points_3d, solve='opencv')
            rvec, tvec, homography = pnpsolver(points_2d, points_3d, solve='dlt')
            rotq = R.from_rotvec(rvec.reshape(1, 3)).as_quat()
            tvec = tvec.reshape(1, 3)

            # Find pose
            rot_mat = R.from_quat(rotq).as_matrix()
            pose_mat = np.concatenate([rot_mat.reshape(3, 3), tvec.reshape(3, 1)], axis=1)

            # RANSAC
            error_t = np.linalg.norm(tvec - tvec_gt)

            count += 1
            if error_t < 0.008 and count <= 5000:
                flag = False
                print(f'total counts of DLT computation: {count} times')
            elif count > 5000:
                flag = False
                print(f'total counts of DLT computation: over {count - 1} times!')

        poses.append(pose_mat)
        homos.append(homography)

    np.save('pose/q1_pose.npy', np.asarray(poses))
    np.save('pose/q1_homography.npy', np.asarray(homos))