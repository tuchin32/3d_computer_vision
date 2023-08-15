import cv2 as cv
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

def reprojection(homo, points, k=8):
    ones = np.ones((k, 1))
    points = np.concatenate((points, ones), axis=1)
    points_proj = np.dot(homo, points.T)
    points_proj[0, :] /= points_proj[2, :]
    points_proj[1, :] /= points_proj[2, :]
    return (points_proj[:2, :]).T

def painter(vertice, surface, camera):
    center = []
    for s in surface:
        v = vertice[s, :]
        center.append((v[0, :] + v[3, :]) / 2)

    depth = np.linalg.norm(center - camera, axis=1)
    order = np.argsort(depth)[::-1]   # from farest to closet
    return order

def make_voxel(vertice, surface, order):
    voxel_set = []
    for ord in order:
        voxel = []
        v = vertice[surface[ord], :]

        for i in range(4):
            voxel.append(list(v[i, :]))

        v02_x = np.linspace(v[0, 0], v[2, 0], 5, endpoint=True)
        v02_y = np.linspace(v[0, 1], v[2, 1], 5, endpoint=True)
        v13_x = np.linspace(v[1, 0], v[3, 0], 5, endpoint=True)
        v13_y = np.linspace(v[1, 1], v[3, 1], 5, endpoint=True)

        for i in range(1, 4):
            voxel.append(list([v02_x[i], v02_y[i]]))
            voxel.append(list([v13_x[i], v13_y[i]]))

        for i in range(5):
            vab_x = np.linspace(voxel[2*i][0], voxel[2*i+1][0], 5, endpoint=True)
            vab_y = np.linspace(voxel[2*i][1], voxel[2*i+1][1], 5, endpoint=True)
            for j in range(1, 4):
                voxel.append([vab_x[j], vab_y[j]])

        voxel_set.append(voxel)

    return voxel_set

def paint_voxel(image, voxel, order, palette):
    image = np.asarray(image)
    rows, cols = image.shape[0], image.shape[1]

    for i in range(len(voxel)):
        for pt in voxel[i]:
            if pt[0] >= 0 and pt[0] < rows and pt[1] >= 0 and pt[1] < cols:
                image = cv.circle(image, (int(pt[0]), int(pt[1])), 20, tuple(palette[order[i]]), -1)
    
    return image

if __name__ == '__main__':
    # Load files
    images_df = pd.read_pickle("data/images.pkl")
    cube_v = np.load('cube/cube_vertices.npy')
    homos = np.load('pose/q1_homography.npy')
    poses = np.load('pose/q1_pose.npy')
    print(f'cube 8 vertices:\n{cube_v}')

    # Create image id list
    num_valid = 130
    start_id = 164
    id = [252, 164, 175, 186, 197, 208, 219, 230, 241,
          253, 264, 275, 286, 288, 289, 290, 291, 292, 293]
    id_list = id

    for j in range(1, 13):
        if j == 8:
            id_list += [i for i in range(id[j] + 1, id[j + 1] - 1)]
        else:
            id_list += [i for i in range(id[j] + 1, id[j + 1])]

    surfaces = [[0, 1, 2, 3], [4, 5, 6, 7],
                [2, 6, 3, 7], [0, 4, 1, 5],
                [2, 6, 0, 4], [3, 7, 1, 5]]
    colors = [[128, 0, 0], [255, 204, 0],
              [128, 128, 0], [0, 255, 128],
              [0, 128, 128], [72, 61, 139]]

    # Capture images and save them as a video
    # fourcc = cv.VideoWriter_fourcc(*'MJPG')
    # out = cv.VideoWriter('q2_output.mp4', fourcc, 10.0, (1080, 1920))

    for imgid in id_list:
        # Load quaery image
        img_name = ((images_df.loc[images_df["IMAGE_ID"] == imgid])["NAME"].values)[0]
        img = cv.imread("data/frames/" + img_name)

        # Load pose
        homo = homos[imgid - start_id, :, :]
        pose = poses[imgid - start_id, :, :]
        camera_ctr = -pose[:, 3]

        # Get the depth order of the cube
        order = painter(cube_v, surfaces, camera_ctr)

        # Prooduce and paint the voxels
        vertice2d = reprojection(homo, cube_v, k=8)
        voxels = make_voxel(vertice2d, surfaces, order)
        img_voxel = paint_voxel(img, voxels, order, colors)

        # Show and then save the image frame
        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.resizeWindow('frame', 360, 640)
        cv.imshow('frame', img_voxel)
        if cv.waitKey(100) == ord('q'):
            break
        # out.write(img)

    # out.release()
    cv.destroyAllWindows()
    # print('Check the result: q2_output.mp4')