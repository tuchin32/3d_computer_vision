from random import sample
import sys
import numpy as np
import cv2 as cv

def get_sift_correspondences(img1, img2):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    pair_info = (kp1, kp2, good_matches)
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return points1, points2, pair_info

def show_pairs(img1, img2, pair_info, indices):
    k = str(len(indices))
    kp1, kp2, matches = pair_info
    matches = np.array(matches)
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, matches[indices], None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.namedWindow('match' + k, cv.WINDOW_NORMAL)
    cv.resizeWindow('match' + k, 1200, 600)
    cv.imshow('match' + k, img_draw_match)
    cv.waitKey(0)
    cv.destroyAllWindows()

def sample_pairs(point, index):
    if len(index) == 1:
        return point[30: 30 + index]
    else:
        return point[index]

def corr_matrix(points1, points2):
    mat_A = []
    for pt1, pt2 in zip(points1, points2):
        u, v = pt1
        x, y = pt2
        mat_A.append([0, 0, 0, u, v, 1, -y * u, -y * v, -y])
        mat_A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
    return mat_A

def direct_linear_transform(points1, points2):
    mat_corr = corr_matrix(points1, points2)
    mat_u, mat_s, mat_vt = np.linalg.svd(mat_corr)
    homography = mat_vt[-1, :].reshape((3, 3))
    return homography / homography[-1, -1]

def reprojection(homo, points, k):
    ones = np.ones((k, 1))
    points = np.concatenate((points, ones), axis=1)
    points_proj = np.dot(homo, points.T)
    points_proj[0, :] /= points_proj[2, :]
    points_proj[1, :] /= points_proj[2, :] 
    return (points_proj[:2, :]).T

def mse(output, target):
    return np.linalg.norm(output - target) / output.shape[0]

def similarity_transform(points):
    # Similarity
    u, v = points[:, 0], points[:, 1]
    mean_u, std_u = np.mean(u), np.std(u)
    mean_v, std_v = np.mean(v), np.std(v)
    similarity =  np.array([[1 / std_u, 0, -mean_u / std_u], 
                            [0, 1 / std_v, -mean_v / std_v],
                            [0, 0, 1]])
    # Normalization
    norm_u = ((u - mean_u) / std_u).reshape((-1, 1))
    norm_v = ((v - mean_v) / std_v).reshape((-1, 1))
    norm_points = np.concatenate((norm_u, norm_v), axis=1)
    return similarity, norm_points

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    
    points1, points2, pair_info = get_sift_correspondences(img1, img2)
    p_s, p_t = gt_correspondences[0], gt_correspondences[1]
    
    num_pts = np.min([len(points1), len(points2)]) # 1-1: 1467, 1-2: 88
    num_tests = 5000

    k_list = [4, 8, 20, 50]
    for k in k_list:
        print('k = %d' % (k))
        records = {'loss': [], 'loss_n': [], 'pairs':[]}

        for _ in range(num_tests):
            # 1-1: Sample the k pairs
            indices = np.random.choice(num_pts, k, replace=False)
            pts1 = sample_pairs(points1, indices)
            pts2 = sample_pairs(points2, indices)

            # 1-2: Implement direct linear transform and compute error
            homography = direct_linear_transform(pts1, pts2)
            p_t_proj = reprojection(homography, p_s, p_s.shape[0])
            error = mse(p_t_proj, p_t)

            # 1-3: Implement direct linear transform and compute error
            sim1, n_pts1 = similarity_transform(pts1)
            sim2, n_pts2 = similarity_transform(pts2)
            n_homography = direct_linear_transform(n_pts1, n_pts2)
            homography = np.dot(np.linalg.inv(sim2), np.dot(n_homography, sim1))
            homography /= homography[-1, -1]
    

            p_t_proj = reprojection(homography, p_s, p_s.shape[0])
            error_n = mse(p_t_proj, p_t)

            # Record the error
            records['loss'].append(error)
            records['loss_n'].append(error_n)
            records['pairs'].append(indices)
        
        # Find proper pairs to get smallest errors
        min_loss = np.argmin(records['loss_n'])
        print('Min error (DLT) = %.6f' % (records['loss'][min_loss]))
        print('Min error (normalized DLT) = %.6f' % (records['loss_n'][min_loss]))
        print('The k-th pairs used in homography estimation:', sorted(records['pairs'][min_loss]), '\n')
        show_pairs(img1, img2, pair_info, records['pairs'][min_loss])