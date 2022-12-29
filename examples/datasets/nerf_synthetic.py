"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from .utils import Rays


def _load_renderings(root_fp: str, subject_id: str, split: str):
    data_dir = os.path.join(root_fp, subject_id)
    
    with open(os.path.join(data_dir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    images = []
    camtoworlds = []
    intrinsics = []
    depths = np.empty([4,0], dtype=float)
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame['file_path'][2:])
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        
        #per image intrinsics
        focal, cx, cy = frame['fl_x'], frame['cx'], frame['cy']
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        intrinsics.append(K)

        #depth images
        depth_path = os.path.join(data_dir,'depth',os.path.basename(fname).split('.')[0]+'.npy')
        pts = np.load(depth_path)
        
        idx = np.ones([1, pts.shape[1]]) * i
        pts = np.vstack((idx,pts))

        depths = np.append(depths,pts,axis=1)

    images = np.stack(images, axis=0) #assume all images have same size
    camtoworlds = np.stack(camtoworlds, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    aabb = meta["aabb"]

    return images, depths, camtoworlds, focal, intrinsics, aabb


class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""

    SPLITS = ["train", "val", "trainval", "test", "None"]
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(   self,subject_id: str,root_fp: str,split: str,color_bkgd_aug: str = "black",
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
        
        self.images, self.depths, self.camtoworlds, self.focal, self.K, self.aabb = _load_renderings(root_fp, subject_id, split)

        self.images = torch.from_numpy(self.images).to(torch.uint8)
        self.depths = torch.from_numpy(self.depths).to(torch.float32)
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.from_numpy(self.K).to(torch.float32)

        self.HEIGHT, self.WIDTH = self.images.shape[1:3]

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
            color_bkgd = torch.rand(3, device=self.images.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.images.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.images.device)

        if rgba.shape[-1] == 4:
            pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        else:
            pixels = rgba

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(0,len(self.images),size=(num_rays,),device=self.images.device)
            else:
                image_id = [index]

            x = torch.randint(0, self.WIDTH, size=(num_rays,), device=self.images.device)
            y = torch.randint(0, self.HEIGHT, size=(num_rays,), device=self.images.device)
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.WIDTH, device=self.images.device),
                torch.arange(self.HEIGHT, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:,0, 2] + 0.5) / K[:,0, 0],
                    (y - K[:,1, 2] + 0.5) / K[:,1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, rgba.shape[1]))
        else:
            origins = torch.reshape(origins, (self.HEIGHT, self.WIDTH, 3))
            viewdirs = torch.reshape(viewdirs, (self.HEIGHT, self.WIDTH, 3))
            rgba = torch.reshape(rgba, (self.HEIGHT, self.WIDTH, rgba.shape[1]))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }

    def get_depth_rays(self):
        num_rays = self.num_rays
        idx = torch.randint(0,self.depths.shape[1],size=(num_rays,),device=self.images.device)

        rand_depths = self.depths[:,idx]
        image_id, x, y, depths = np.vsplit(rand_depths, 4)

        image_id = image_id.reshape([num_rays]).to(torch.long)
        x = x.reshape([num_rays]).to(torch.long)
        y = y.reshape([num_rays]).to(torch.long)

        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        K = self.K[image_id]
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - K[:,0, 2] + 0.5) / K[:,0, 0],
                    (y - K[:,1, 2] + 0.5) / K[:,1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        origins = torch.reshape(origins, (num_rays, 3))
        viewdirs = torch.reshape(viewdirs, (num_rays, 3))
        depths = torch.reshape(depths, (num_rays, 1))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        if self.color_bkgd_aug == "random":
            color_bkgd = torch.rand(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "white":
            color_bkgd = torch.ones(3, device=self.camtoworlds.device)
        elif self.color_bkgd_aug == "black":
            color_bkgd = torch.zeros(3, device=self.camtoworlds.device)

        return {
            "depths": depths,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
            "color_bkgd": color_bkgd
        }