import sys
import math
import numpy as np
import cv2 as cv

def set_mouse(event, x, y, flags, data):
    image, points = data

    if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
        # Mark 4 corners
        cv.circle(image, (x, y), 25, (0, 0, 255), -1)
        cv.circle(image, (x, y), 25, (0, 0, 0), 5)

        cv.imshow('Mark four corners', image)

        print('Mark the point: (h, w) = (%d, %d)' % (y, x))
        points.append([y, x])

        if len(points) == 4:
            print('End of marking! Press any key to colse the window.')

def get_corners(image):
    corners = []
    cor_image = image.copy()

    cv.namedWindow('Mark four corners', cv.WINDOW_NORMAL)
    cv.resizeWindow('Mark four corners', 1200, 900)
    cv.imshow('Mark four corners', image)
    
    print('Please mark four corners of the currency.')
    print('Order: 1.left-top 2.right-top 3.left-bottom 4.right-bottom.')
    cv.setMouseCallback('Mark four corners', set_mouse, (cor_image, corners))
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('./images/2-1.jpg', cor_image)
    return np.array(corners)

def get_references(image, gbp_size):
    h, w = image.shape[0], image.shape[1]
    gbp_h, gbp_w = gbp_size

    scale_h = h / gbp_h
    scale_w = w / gbp_w

    if scale_h > scale_w:
        new_h = math.floor(gbp_h * scale_w)
        new_w = math.floor(gbp_w * scale_w)
    else:
        new_h = math.floor(gbp_h * scale_h)
        new_w = math.floor(gbp_w * scale_h)

    return np.zeros([new_h, new_w, 3]), \
           np.array([[0, 0], [0, new_w - 1], [new_h - 1, 0], [new_h - 1, new_w - 1]])

def corr_matrix(points1, points2):
    mat_A = []
    for pt1, pt2 in zip(points1, points2):
        u, v = pt1
        x, y = pt2
        mat_A.append([0, 0, 0, -u, -v, -1, y * u, y * v, y])
        mat_A.append([u, v, 1, 0, 0, 0, -x * u, -x * v, -x])
    return mat_A

def direct_linear_transform(points1, points2):
    mat_corr = corr_matrix(points1, points2)
    mat_u, mat_s, mat_vt = np.linalg.svd(mat_corr)
    homography = mat_vt[-1, :].reshape((3, 3))
    return homography / homography[-1, -1]

def projection(homo, points, k):
    ones = np.ones((k, 1))
    points = np.concatenate((points, ones), axis=1)
    points_proj = np.dot(homo, points.T)
    points_proj[0, :] /= points_proj[2, :]
    points_proj[1, :] /= points_proj[2, :] 
    return (points_proj[:2, :]).T

def warping(image, new_image, homo):
    rows, cols = new_image.shape[0], new_image.shape[1]
    new_h = np.arange(0, rows, 1)
    new_w = np.arange(0, cols, 1)
    new_hh, new_ww = np.meshgrid(new_h, new_w)
    new_hw = np.array([new_hh.flatten(), new_ww.flatten()]).T

    points = projection(homo, new_hw, new_hw.shape[0])
    for i in range(points.shape[0]):
        h, w = points[i, 0], points[i, 1]
        floor_h = math.floor(h)
        floor_w = math.floor(w)
        dh = h - floor_h
        dw = w - floor_w
        new_image[new_hw[i, 0], new_hw[i, 1]] = (1 - dh) * (1 - dw) * image[floor_h, floor_w] \
                                              + dh * (1 - dw) * image[floor_h + 1, floor_w] \
                                              + (1 - dh) * dw * image[floor_h, floor_w + 1] \
                                              + dh * dw * image[floor_h + 1, floor_w + 1]


if __name__ == '__main__':
    # 1. Capture document
    img = cv.imread(sys.argv[1])
    img = np.array(img)

    # 2. Mark 4 corner points
    corners = get_corners(img)
    # corners = np.array([[475, 596], [493, 1672], [987, 468], [1050, 1765]])

    # 3. Compute homography
    gbp_20 = (73, 139)
    ref_img, ref_pts = get_references(img, gbp_20)
    homography = direct_linear_transform(ref_pts, corners)

    # 4. Bilinear warping
    warping(img, ref_img, homography)
    cv.imwrite('./images/2-2.jpg', ref_img)
    print('Check the result: ./images/2-2.jpg.')

    
    