"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os

import einops as eo
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from read_write_model import *

from .utils import Rays, depth_Rays
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
    
    aabb = [[-1, -1, -1], [1, 1, 1]] # Could possible be determined programatically 
    # -------------------------------------------------------------------------------------------
    # ---Load colmap data, need .bin version of colmap outputs-----------------------------------
    # -------------------------------------------------------------------------------------------

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0")
    col_images = read_images_binary(os.path.join(colmap_dir, "images.bin"))
    col_points = read_points3d_binary(os.path.join(colmap_dir, "points3D.bin"))
    col_cameras = read_cameras_binary(os.path.join(colmap_dir, "cameras.bin"))

    proj_errs = np.array([point3D.error for point3D in col_points.values()])
    proj_errs_mean = np.mean(proj_errs)
    print("Mean Projection Error:", proj_errs_mean)
    
    # -------------------------------------------------------------------------------------------
    # ---Colmap data (opencv, relative first image) to nerf (opengl, rotated and centered)-------
    # -------------------------------------------------------------------------------------------

    images = [] # image matrix (N, W, H)
    camtoworlds = [] # Poses (N, 4, 4)
    intrinsics = [] # Instrinsics (N, 3, 3) [[focal, 0, cx], [0, focal, cy], [0, 0, 1]]
    up = np.zeros(3)
    for i in col_images:
        col_image = col_images[i]
        image_path = os.path.join(data_dir, "images/" + col_image.name)
        if os.path.splitext(col_image.name)[-1] == '.png':
            images.append(imageio.imread(image_path, ignoregamma=True))
        else:
            images.append(imageio.imread(image_path))

        R = col_image.qvec2rotmat()
        t = col_image.tvec.reshape([3,1])
        bottom = np.array([0,0,0,1.]).reshape([1,4])
        w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        c2w = np.linalg.inv(w2c)
        c2w = c2w @ np.diag([1, -1, -1, 1]) # opencv to opengl
        c2w = c2w[[1,0,2,3],:] # swap y and z
        c2w[2,:] *= -1 # flip whole world upside down
        up += c2w[0:3,1]

        camtoworlds.append(c2w)
        focal, cx, cy, distortion = col_cameras[col_image.camera_id].params
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)
    
    up /= np.linalg.norm(up)
    print("up vector was", up)
    R = rotmat(up,[0, 0, 1]) # rotate up vector to [0,0,1]
    R = np.pad(R,[0, 1])
    R[-1, -1] = 1
    # images = np.stack(images, axis=0) #assume all images have same size
    intrinsics = np.stack(intrinsics, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    camtoworlds = R @ camtoworlds # (N, 4, 4)
    
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

    # -------------------------------------------------------------------------------------------
    # ---Obtain sparse points for depth supervision----------------------------------------------
    # -------------------------------------------------------------------------------------------

    depth_list = [] # Sparse point depths (N, points)
    coord_list = [] # Sparse point pixel locations in image [N, points, X, Y]
    weight_list = [] # Reprojection Errors (N, points)
    size_list = [] # Number of points in each image
    for i, im_id in enumerate(col_images):
        col_image = col_images[im_id]
        pts3D_id = col_image.point3D_ids[col_image.point3D_ids > -1]
        coord_list.append(col_image.xys[col_image.point3D_ids > -1])
        size_list.append(len(pts3D_id))
        c2w = camtoworlds[i]

        depth, point3D, errs = [], [], []
        for id in pts3D_id:
            points = R[:-1, :-1] @ col_points[id].xyz # rotate 3D sparse point
            depth.append(c2w[:3, 2].T @ (points - c2w[:3 ,3]))
            point3D.append(points)
            errs.append(col_points[id].error)

        depth_list.append(np.stack(depth))
        weight_list.append(2 * np.exp(-(errs/proj_errs_mean)**2))
    

    depth_gts = {"depth":np.array(depth_list), "coord":np.array(coord_list), "weight":np.array(weight_list), "size": size_list}
    return images, camtoworlds, focal, intrinsics, aabb, depth_gts

def get_rays_by_coord_np(H, W, focal, c2w, coords):
    i, j = (coords[:,0]-W*0.5)/focal, -(coords[:,1]-H*0.5)/focal
    dirs = torch.stack([i,j,-torch.ones_like(i)],-1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = torch.broadcast_to(c2w[:3,-1], rays_d.shape)
    return rays_o, rays_d

def get_depth_rays(H, W, focal, poses, depth_gts):
    rays_depth = None
    print('get depth rays')
    rays_depth_list = []
    for i, _ in enumerate(i_train):
        rays_depth = np.stack(get_rays_by_coord_np(H, W, focal, poses[i,:3,:4], depth_gts[i]['coord']), axis=0) # 2 x N x 3 [1,N,3: rays_d, 2,N,3: rays_o]
        # print(rays_depth.shape)
        rays_depth = eo.rearrange(rays_depth, 'rays N xys -> N rays xys')
        rays_depth = np.transpose(rays_depth, [1,0,2])
        depth_value = np.repeat(depth_gts[i]['depth'][:,None,None], 3, axis=2) # N x 1 x 3
        weights = np.repeat(depth_gts[i]['weight'][:,None,None], 3, axis=2) # N x 1 x 3
        rays_depth = np.concatenate([rays_depth, depth_value, weights], axis=1) # N x 4 x 3
        rays_depth_list.append(rays_depth)

    rays_depth = np.concatenate(rays_depth_list, axis=0)
    print('rays_weights mean:', np.mean(rays_depth[:,3,0]))
    print('rays_weights std:', np.std(rays_depth[:,3,0]))
    print('rays_weights max:', np.max(rays_depth[:,3,0]))
    print('rays_weights min:', np.min(rays_depth[:,3,0]))
    print('rays_depth.shape:', rays_depth.shape)
    rays_depth = rays_depth.astype(np.float32)
    print('shuffle depth rays')
    np.random.shuffle(rays_depth)

    max_depth = np.max(rays_depth[:,3,0])
    print('done')
    i_batch = 0

def getRays(img, x, y, K, c2w, OPENGL_CAMERA=True):
    rgba = img[x, y] / 255.0   # (num_rays, 4)

    camera_dirs = F.pad(
        torch.stack(
            [
                (y - K[0, 2] + 0.5) / K[0, 0],
                (x - K[1, 2] + 0.5) / K[1, 1] * (-1.0 if OPENGL_CAMERA else 1.0),
            ],
            dim=-1,
        ),
        (0, 1),
        value=(-1.0 if OPENGL_CAMERA else 1.0),
    )  # [num_rays, 3]
    
    # [n_cams, height, width, 3]
    directions = (camera_dirs[:, None, :] * c2w[:3, :3]).sum(dim=-1)
    ray_o = torch.broadcast_to(c2w[:3, -1], directions.shape)
    ray_d = directions / torch.linalg.norm(
        directions, dim=-1, keepdims=True
    )
    return ray_o, ray_d, rgba

class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test", "None"]
    NEAR, FAR = 0.0, 6.0
    OPENGL_CAMERA = True

    def __init__(   self,subject_id: str,root_fp: str,split: str,color_bkgd_aug: str = "random",
                    num_rays: int = None,near: float = None,far: float = None,batch_over_images: bool = True):

        super().__init__()
        assert split in self.SPLITS, "%s" % split
        assert color_bkgd_aug in ["white", "black", "random"]
        
        self.split = split
        self.num_rays = num_rays
        
        self.near = self.NEAR if near is None else near
        self.far = self.FAR if far is None else far
        
        self.training = (num_rays is not None) and (split in ["train", "None"])
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        
        self.images, self.camtoworlds, self.focal, self.K, self.aabb, self.depth_gts = _load_renderings(root_fp, subject_id, split)


        # self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.from_numpy(self.K).to(torch.float32)

        # self.HEIGHT, self.WIDTH = self.images.shape[1:3]

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        
        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.camtoworlds.device)

        if rgba.shape[-1] == 4:
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        else:
            pixels = rgba

        dep_rgba, dep_rays = data["depth_rgba"], data["depth_rays"]
        if dep_rgba.shape[-1] == 4:
            dep_pixels, alpha = torch.split(dep_rgba, [3, 1], dim=-1)
            dep_pixels = dep_pixels * alpha + color_bkgd * (1.0 - alpha)
        else:
            dep_pixels = dep_rgba

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "depth_pixels": dep_pixels,  # [n_rays, 3] or [h, w, 3]
            "depth_rays": dep_rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays", "depth_rgba", "depth_rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            
            rgba_ray_o = []
            rgba_ray_d = []
            rgba_rgba = []
            dep_ray_o = []
            dep_ray_d = []
            dep_rgba = []
            dep_depth = []
            dep_weight = []
            depth_prop = 0.5
            N = len(self.images)
            tot_pts = sum(self.depth_gts['size'])
            pts_used = 0
            for i in range(N):
                img = torch.from_numpy(self.images[i]).to(torch.uint8).to(device=self.camtoworlds.device)
                
                # Get rgba rays
                H, W, _ = img.shape
                num_color_rays = int(np.round(num_rays*(1-depth_prop)))
                if i == N-1:
                    rgba_batch = num_color_rays-torch.cat(rgba_rgba,0).shape[0]
                else: 
                    rgba_batch = num_color_rays//N

                xc = torch.randint(0, H, size=(rgba_batch,), device=self.camtoworlds.device)
                yc = torch.randint(0, W, size=(rgba_batch,), device=self.camtoworlds.device)
                c2w = self.camtoworlds[i]  # (num_rays, 3, 4)
                ray_o, ray_d, rgba = getRays(img, xc, yc, self.K[i], c2w)
                rgba_ray_o.append(ray_o)
                rgba_ray_d.append(ray_d)
                rgba_rgba.append(rgba)
                
                # Get depth rays
                num_dep_rays = num_rays-num_color_rays

                if i == N-1:
                    pts_to_use = num_dep_rays - pts_used
                else:
                    pts_to_use = int(round(self.depth_gts['size'][i]/tot_pts*num_dep_rays, 0))

                pts_used += pts_to_use
                pts_i = torch.randint(0, self.depth_gts['size'][i]-1, size=(pts_to_use,))
                xy = torch.from_numpy(np.round(self.depth_gts['coord'][i][pts_i])).to(torch.long).to(device=self.camtoworlds.device)
                y = xy[:, 0]
                x = xy[:, 1]
                
                depth = torch.from_numpy(self.depth_gts['depth'][i][pts_i]).to(torch.float).to(device=self.camtoworlds.device)
                weight = torch.from_numpy(self.depth_gts['weight'][i][pts_i]).to(torch.float).to(device=self.camtoworlds.device)

                ray_o, ray_d, rgba = getRays(img, x, y, self.K[i], c2w)
                dep_ray_o.append(ray_o)
                dep_ray_d.append(ray_d)
                dep_rgba.append(rgba)
                dep_depth.append(depth)
                dep_weight.append(weight)

        rgba_ray_o = torch.cat(rgba_ray_o,0)
        rgba_ray_d = torch.cat(rgba_ray_d,0)
        rgba_rgba = torch.cat(rgba_rgba,0)
        dep_ray_o = torch.cat(dep_ray_o,0)
        dep_ray_d = torch.cat(dep_ray_d,0)
        dep_rgba = torch.cat(dep_rgba,0)
        dep_depth = torch.cat(dep_depth,0)
        dep_weight = torch.cat(dep_weight,0)

        rays = Rays(origins=rgba_ray_o, viewdirs=rgba_ray_d)
        depth_rays = depth_Rays(origins=dep_ray_o, viewdirs=dep_ray_d, depths=dep_depth, weights=dep_weight)

        return {
            "rgba": rgba_rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "depth_rgba": dep_rgba, 
            "depth_rays": depth_rays,
        }
