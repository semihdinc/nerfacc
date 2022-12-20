"""
This script is for testing analyzing rays in ENU coordinate space
@author: semih dinc (11/10/2022)
"""

import json
import os
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pdb
from scipy.spatial.transform import Rotation as SR

def get_rays_np(H, W, K, c2w):
   i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
   dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
   # Rotate ray directions from camera frame to the world frame
   rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
   # Translate camera frame's origin to the world frame. It is the origin of all rays.
   rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
   return rays_o, rays_d

def intrinsics(frame):
   f = frame["fl_x"]
   cx, cy = frame["cx"], frame["cy"]
   W, H = frame["w"], frame["h"]
   K = np.array([[f, 0, cx],[0, f, cy], [0, 0, 1]])
   return int(H),int(W),K

#generates "numberOfFrames" poses in spherical position
#each pose is looking at 0,0,0 and rotates around Z axis
def generateTestPoses(p0, numberOfFrames=30):
   # transMat = np.array([[1,0,0,t[0]],[0,1,0,t[1]],[0,0,1,t[2]],[0,0,0,1]]).astype(float)
   # rotMatX = np.identity(4)
   # rotMatX[0:3,0:3] = SR.from_euler('X',-np.pi/4).as_matrix()
   poses = []
   for angle in np.linspace(0,np.pi,numberOfFrames):

      rotMatZ = np.identity(4)
      rotMatZ[0:3,0:3] = SR.from_euler('Z',angle).as_matrix()

      myPose = rotMatZ @ p0
      poses.append(myPose)

   return poses

def analyzeRays(data_dir,show_rays=False):
   
   poses = []
   selected_rays = []
   with open(os.path.join(data_dir,"transforms.json")) as jsonFile:
      metadata = json.load(jsonFile)
      
      all_lengths = np.zeros((len(metadata['frames']),5))
      for i, frame in enumerate(metadata['frames']):
         H,W,K = intrinsics(frame)
         c2w = np.array(frame['transform_matrix'])
         poses.append(c2w)

         #generate rays from the poses
         rays_o, rays_d = get_rays_np(H, W, K, c2w)
         
         selected_rays = []
         selected_rays.append(rays_d[0,0])     #top left
         selected_rays.append(rays_d[H-1,0])   #bottom left
         selected_rays.append(rays_d[0,W-1])   #top right
         selected_rays.append(rays_d[H-1,W-1]) #bottom right
         selected_rays.append(rays_d[int(H/2),int(W/2)])   #center ray

         ray_o = c2w[0:3,3]

         ray_lengths = []
         for ray in selected_rays:
            ray = ray / np.linalg.norm(ray) #normalize the ray length to unit vector (1 meter)
            
            near, far = ray_o[2], 0 #I want my ray start from ray_o and go down to the ground (~0 meters)

            #find how many times unit vector must be multiplied to reach near and far altitudes
            near = (near-ray_o[2])/ray[2]
            far = (far-ray_o[2])/ray[2]

            ray_lengths.append(far)

            #test to see if ray is actually going to desired altitudes (near=ray_o[2] and far=0)
            ray_start = ray_o + near * ray
            ray_end = ray_o + far * ray

         all_lengths[i] = np.array(ray_lengths)
   a = 1
   
def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Camera Positions")
    parser.add_argument("--data_dir", default="", help="Input path to the EVIW images and transform.json")
    parser.add_argument("--show_rays", action="store_true", help="show the center ray of each pose")
    return parser.parse_args()

if __name__ == '__main__':
   np.set_printoptions(suppress=True)
   args = parse_args()
   args.data_dir = "/Users/semih.dinc/Desktop/IW_Imagesets/shuttle_enu_0_10/"
   analyzeRays(args.data_dir, args.show_rays)
   