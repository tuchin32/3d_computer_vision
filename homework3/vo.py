import open3d as o3d
import numpy as np
import cv2 as cv
import sys, os, argparse, glob
import multiprocessing as mp

class SimpleVO:
    def __init__(self, args):
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']
        
        self.frame_paths = sorted(list(glob.glob(os.path.join(args.input, '*.png'))))
        
        self.artificial = True
        self.color_step = 0

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    pyramid = self.load_pyramid(R, t)
                    vis.add_geometry(pyramid)
            except: pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

    def load_pyramid(self, rot_mat, t_vec):
        # Inverse the pose to get the camera pose
        r_cam = np.linalg.inv(rot_mat)
        t_cam = -np.dot(r_cam, t_vec.reshape(3))

        # Find the three camera axes in the world coordinate system
        axes_c = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        axes_w = np.dot(np.linalg.inv(r_cam), axes_c.T)

        # Create the pyramid
        xw = axes_w[:, 0] / np.linalg.norm(axes_w[:, 0])
        yw = axes_w[:, 1] / np.linalg.norm(axes_w[:, 1])
        zw = 3 * axes_w[:, 2] / np.linalg.norm(axes_w[:, 2])

        center = -np.dot(np.linalg.inv(r_cam), t_cam.reshape(3))
        ctr_proj = center - zw

        base_w0 = ctr_proj + xw + yw
        base_w1 = ctr_proj + xw - yw
        base_w2 = ctr_proj - xw + yw
        base_w3 = ctr_proj - xw - yw

        # Use Open3D to draw the pyramid
        points = [center, base_w0, base_w1, base_w2, base_w3]
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]
        colors = [[0.00156 * self.color_step, 0.00198 * self.color_step, 0.00177 * self.color_step]
                   for i in range(len(lines))]
        self.color_step += 1

        pyramid = o3d.geometry.LineSet()
        pyramid.points = o3d.utility.Vector3dVector(points)
        pyramid.lines  = o3d.utility.Vector2iVector(lines)
        pyramid.colors = o3d.utility.Vector3dVector(colors)

        return pyramid


    def process_frames(self, queue):
        R, t = np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64)
        buff = {'pts': None, 'tri_pts': None, 't': None}
        num_frames = len(self.frame_paths[1:])

        for idx in range(num_frames):
            print(f'Processing frame {idx + 1} / {num_frames}')

            # 1. Capture new frame
            img = cv.imread(self.frame_paths[idx + 1])
            img_old = cv.imread(self.frame_paths[idx])

            # 2. Extract and match features between the two images
            pts1, pts2 = self.orb_bfmatching(img_old, img)

            # Undistort points
            h, w = img.shape[:2]
            new_K, roi = cv.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), 1, (w, h))
            pts1 = cv.undistortPoints(pts1.reshape(-1, 1, 2), self.K, self.dist, P=new_K).reshape(-1, 2)
            pts2 = cv.undistortPoints(pts2.reshape(-1, 1, 2), self.K, self.dist, P=new_K).reshape(-1, 2)
            
            # 3. Find the Essential Matrix
            E, mask = cv.findEssentialMat(pts1, pts2, self.K, method=cv.RANSAC, prob=0.999, threshold=1)

            # 4. Decompose the Essential Matrix to find the camera pose
            _, R_rel, t_rel, mask, tri_pts = cv.recoverPose(E, pts1, pts2, self.K, distanceThresh=1000, mask=mask)
            
            tri_pts = (tri_pts / tri_pts[3]).T
            P_rel = np.concatenate((R_rel, t_rel), axis=1)
            P_rel = np.concatenate((P_rel, np.array([[0, 0, 0, 1]])), axis=0)

            if idx == 0:
                # 6. Compute the cumulative pose
                P_cum = np.copy(P_rel)

            else:
                if self.artificial == True:
                    if idx == 55:
                        deg = 10
                        R_rel = np.dot(R_rel, np.array([[1, 0, 0],
                                                        [0, np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
                                                        [0, np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]]))
                    elif idx == 220 or idx == 230:
                        # Rotate 10 degrees around the x-axis
                        deg = -10
                        R_rel = np.dot(R_rel, np.array([[1, 0, 0],
                                                        [0, np.cos(np.deg2rad(deg)), -np.sin(np.deg2rad(deg))],
                                                        [0, np.sin(np.deg2rad(deg)), np.cos(np.deg2rad(deg))]]))
                    elif idx == 286 or idx == 287:
                        # Rotate -15 dedgrees around the y-axis
                        deg = -15
                        R_rel = np.dot(R_rel, np.array([[np.cos(np.deg2rad(deg)), 0, np.sin(np.deg2rad(deg))],
                                                        [0, 1, 0],
                                                        [-np.sin(np.deg2rad(deg)), 0, np.cos(np.deg2rad(deg))]]))
                    P_rel[:3, :3] = np.copy(R_rel)
                
                # 5. Compute scale from previous information
                index = self.coord_matching(pts1, buff['pts'], threshold=len(pts1)//10)
                scale = self.get_scale(buff['tri_pts'], tri_pts, buff['t'], index)
                t_rel = t_rel * scale
                P_rel[:3, 3:] = np.copy(t_rel)

                # 6. Compute the cumulative pose
                P_cum = np.dot(P_cum, P_rel)

            # Store the information
            buff['pts'] = pts2
            buff['tri_pts'] = tri_pts
            buff['t'] = t_rel
            
            # Put the new pose into the queue
            R_cum = np.copy(P_cum[:3, :3])
            t_cum = np.copy(P_cum[:3, 3:])
            queue.put((R_cum, t_cum))
             
            for i in range(len(pts2)):
                cv.circle(img, (int(pts2[i][0]), int(pts2[i][1])), 2, (187, 123, 0), -1)
            cv.imshow('frame', img)
            if cv.waitKey(30) == 27: break
        queue.put((None, None))


    # ORB discriptors and Brute-Force Matching
    def orb_bfmatching(self, img1, img2):
        '''img1: queryImage, img2: trainImage'''

        # Initiate ORB detector
        orb = cv.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # Find good matched points.
        pts1, pts2 = [], []
        for m in matches:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
        pts1, pts2 = np.asarray(pts1), np.asarray(pts2)
        return pts1, pts2

    def coord_matching(self, pts1, pts2, threshold=2):
        index = []
        for i in range(pts1.shape[0]):
            for j in range(pts2.shape[0]):
                diff = np.linalg.norm(pts1[i, :] - pts2[j, :])
                if diff == 0:
                    index.append([i, j])
                if len(index) >= threshold:
                    break
        return np.array(index)

    def get_scale(self, pts3d_old, pts3d, t_old, index):
        scales = []
        for i in range(0, len(index), 2):
            if (i + 1) > (len(index) - 1):
                break
            else:
                id0, id1 = index[i], index[i + 1]
                num = np.linalg.norm(pts3d[id0[0], :] - pts3d[id1[0], :])
                den = np.linalg.norm(pts3d_old[id0[1], :] - pts3d_old[id1[1], :])
                if den == 0:
                    continue
                scales.append(num / den)

        t_mag = np.linalg.norm(t_old)
        scale = np.median(scales) * t_mag

        if scale > 2 or scale < 0.5:
            scale = 1 
        return scale
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()
