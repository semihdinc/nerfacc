import os
import numpy as np
import imageio.v2 as imageio
import json
from read_write_model import *
import cv2

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = cv2.Laplacian(gray, cv2.CV_64F).var()
	return fm

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom

def _load_renderings(root_fp: str, subject_id: str, split: str):
    
    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0")
    
    col_images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
    col_points = read_points3d_binary(os.path.join(colmap_dir, "points3D.bin"))
    col_cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))

    proj_errs = np.array([point3D.error for point3D in col_points.values()])
    proj_errs_mean = np.mean(proj_errs)
    print("Mean Projection Error:", proj_errs_mean)

    images = []
    camtoworlds = []
    intrinsics = []
    depth_list, coord_list, weight_list = [], [], []
    image_paths = []
    sharp = []
    up = np.zeros(3)
    for i in col_images:
        col_image = col_images[i]
        image_paths.append("./images/" + col_image.name)
        
        sharp.append(sharpness(os.path.join(data_dir, "images/" + col_image.name)))
        '''image_path = os.path.join(data_dir, "images/" + col_image.name)
        if os.path.splitext(col_image.name)[-1] == '.png':
            images.append(imageio.imread(image_path, ignoregamma=True))
        else:
            images.append(imageio.imread(image_path))'''

        pts3D_id = col_image.point3D_ids[col_image.point3D_ids > -1]
        coord_list.append(col_image.xys[col_image.point3D_ids > -1])

        R = col_image.qvec2rotmat()
        t = col_image.tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2w = c2w @ np.diag([1, -1, -1, 1])
        c2w = c2w[[1,0,2,3],:] # swap y and z
        c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]

        camtoworlds.append(c2w)
        focal, cx, cy, distortion = col_cameras[col_image.camera_id].params
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)
        depth, point3D, errs = [], [], []
        for i in pts3D_id:
            points = col_points[i].xyz
            depth.append(c2w[:3, 2].T @ (points - c2w[:3 ,3]))
            point3D.append(points)
            errs.append(col_points[i].error)

        depth_list.append(depth)
        weight_list.append(2 * np.exp(-(errs/proj_errs_mean)**2))

    up = up / np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0,0,1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0,1])
    R[-1, -1] = 1
    # images = np.stack(images, axis=0) #assume all images have same size
    camtoworlds = np.stack(camtoworlds, axis=0)
    camtoworlds = R @ camtoworlds
    
    print("computing center of attention...")
    totw = 0.0
    totp = np.array([0.0, 0.0, 0.0])
    for f in camtoworlds:
        for g in camtoworlds:
            p, w = closest_point_2_lines(f[:-1, 3], f[:-1, 2], g[:-1, 3], g[:-1, 2])
            if w > 0.00001:
                totp += p*w
                totw += w
    if totw > 0.0:
        totp /= totw
    print(totp)

    camtoworlds[:, 0:3, 3] -= totp
    
    # Instant-ngp scaling factor
    # avglen = sum([np.linalg.norm(f) for f in camtoworlds[:, 0:3, 3]])/camtoworlds.shape[0]
    # camtoworlds[:, 0:3,3] *= 4.0 / avglen # scale to "nerf sized"

    intrinsics = np.stack(intrinsics, axis=0)
    #poses = camtoworlds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0]) # 3 x 5 x N
    #bds = camtoworlds[:, -2:].transpose([1,0])
    aabb = [[0, 0, 0], [1, 1, 1]]
    depth_gts = 0 #{"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list)}
    return image_paths, camtoworlds, col_cameras, sharp

def saveTransforms(out_fp, image_paths, c2w, cam, sharp):
    meta = {'aabb_scale': 1,
            'scale': 1,
            'aabb': [[-1,-1,-1], [1,1,1]],
            'frames': []}
    
    for i, c2w_i in enumerate(c2w):
        meta['frames'].append({'w': int(cam[1].width),
                'h': int(cam[1].height),
                'fl_x': float(cam[1].params[0]),
                'fl_y': float(cam[1].params[0]),
                'camera_angle_x': np.tanh(cam[1].width / (cam[1].params[0] * 2)) * 2,
                'camera_angle_y': np.tanh(cam[1].height / (cam[1].params[0] * 2)) * 2,
                'k1': float(cam[1].params[3]),
                'k2': 0.0,
                'p1': 0.0,
                'p2': 0.0,
                'cx': float(cam[1].params[1]),
                'cy': float(cam[1].params[2]),
                "file_path": image_paths[i],
                "sharpness": sharp[i],
                'transform_matrix': (c2w_i).tolist() #opencv -> opengl
            } 
        )
    
    with open(out_fp, 'w') as out:
        json.dump(meta, out, indent=2)


if __name__ == "__main__":
    root_fp = "/home/ubuntu/ws/data/nerf"
    subject_id = "shuttleTest8png"
    split = "train"
    out_fp = os.path.join(root_fp, subject_id + "/transforms.json")
    image_paths, c2w, col_cam, sharp = _load_renderings(root_fp, subject_id, split)
    saveTransforms(out_fp, image_paths, c2w, col_cam, sharp)

