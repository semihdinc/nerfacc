import argparse
import torch
import numpy as np
import time
import math
import os
import glob
import cv2

import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed
from nerfacc import ContractionType, OccupancyGrid

import warnings; warnings.filterwarnings("ignore")

class OrbitCamera:
    def __init__(self, K, w, h, r):
        self.K = K
        self.W, self.H = w, h
        self.radius = r
        self.center = np.zeros(3)
        self.rot = np.eye(3)

    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4)
        res[2, 3] += self.radius
        # rotate
        rot = np.eye(4)
        rot[:3, :3] = self.rot
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res

    def orbit(self, dx, dy):
        rotvec_x = self.rot[:, 1] * np.radians(0.05 * dx)
        rotvec_y = self.rot[:, 0] * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_y).as_matrix() @ \
                   R.from_rotvec(rotvec_x).as_matrix() @ \
                   self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        self.center += 1e-4 * self.rot @ np.array([dx, dy, dz])


class NGPGUI:
    def __init__(self, radiance_field, occupancy_grid, render_aabb, testPoses, render_step_size, cone_angle, alpha_thre, radius=2500):

        self.radiance_field = radiance_field
        self.occupancy_grid = occupancy_grid
        self.render_aabb = render_aabb
        self.render_step_size = render_step_size
        self.cone_angle = cone_angle
        self.alpha_thre = alpha_thre

        self.K, self.W, self.H = testPoses.K, testPoses.WIDTH, testPoses.HEIGHT
        self.testPoses = testPoses

        self.cam = OrbitCamera(self.K, self.W, self.H, r=radius)

        self.render_buffer = np.ones((self.W, self.H, 3), dtype=np.float32)

        # placeholders
        self.dt = 0
        self.mean_samples = 0
        self.img_mode = 0

        self.register_dpg()

    def render_cam(self, cam):
        t = time.time()

        c2w = torch.tensor(cam.pose,dtype=torch.float,device='cuda:0')

        data = self.testPoses.get_rays(c2w)
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]

        self.radiance_field.eval()
        with torch.no_grad():
            # rendering
            rgb, acc, depth, _ = render_image(
                self.radiance_field,
                self.occupancy_grid,
                rays,
                self.render_aabb,
                render_step_size = self.render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=self.cone_angle,
                alpha_thre=self.alpha_thre,
                # test options
                test_chunk_size=args.test_chunk_size,
            )
        # save rgb image
        rgbImage = rgb.cpu().numpy()

        cmd = dpg.get_value('colormap_depth')
        depthImage = depth.cpu().numpy()
        depthImage = depthImage[...,-1]
        depthImage[depthImage < cmd] = cmd
        depthImage = (depthImage-depthImage.min())/(depthImage.max()-depthImage.min())
        depthImage = cv2.applyColorMap((depthImage*255).astype(np.uint8), cv2.COLORMAP_TURBO)

        torch.cuda.synchronize()
        self.dt = time.time()-t

        if self.img_mode == 0:
            return rgbImage
        elif self.img_mode == 1:
            return depthImage.astype(np.float32)/255.0

    def register_dpg(self):
        dpg.create_context()
        dpg.create_viewport(title="nerfacc", width=self.W, height=self.H, resizable=False)

        ## register texture ##
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.render_buffer,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture")

        ## register window ##
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):
            dpg.add_image("_texture")
        dpg.set_primary_window("_primary_window", True)

        def callback_depth(sender, app_data):
            self.img_mode = 1-self.img_mode

        ## control window ##
        with dpg.window(label="Control", tag="_control_window", width=200, height=150):
            dpg.add_slider_float(label="min depth", default_value=0,min_value=1500, max_value=3000, tag="colormap_depth")
            dpg.add_button(label="depth/color", tag="_button_depth",callback=callback_depth)
            dpg.add_separator()
            dpg.add_text('no data', tag="_log_time")
            dpg.add_text('no data', tag="_samples_per_ray")

        ## register camera handler ##
        def callback_camera_drag_rotate(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.orbit(app_data[1], app_data[2])

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.scale(app_data)

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.cam.pan(app_data[1], app_data[2])

        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        ## Avoid scroll bar in the window ##
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        ## Launch the gui ##
        dpg.setup_dearpygui()
        dpg.set_viewport_small_icon("assets/icon.png")
        dpg.set_viewport_large_icon("assets/icon.png")
        dpg.show_viewport()

    def render(self):
        while dpg.is_dearpygui_running():
            dpg.set_value("_texture", self.render_cam(self.cam))
            dpg.set_value("_log_time", f'Render time: {1000*self.dt:.2f} ms')
            dpg.set_value("_samples_per_ray", f'Samples/ray: {self.mean_samples:.2f}')
            dpg.render_dearpygui_frame()


if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name",type=str,default="exp",help="The name of the folder for saving results")
    parser.add_argument("--scene",type=str,default="lego",help="which scene to use")
    parser.add_argument("--test_chunk_size",type=int,default=8192)
    parser.add_argument("--max_steps",type=int,default=30000, help="Max number of training iterations")
    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--i_ckpt",type=int, default=1000, help="Iterations to save model")
    parser.add_argument("--render_only",action="store_true",help="whether to only render images")

    #currently useless options
    parser.add_argument("--i_test",type=int, default=5000, help="Iterations to render test poses and create video") 
    parser.add_argument("--aabb",type=lambda s: [float(item) for item in s.split(",")],default="-1.5,-1.5,-1.5,1.5,1.5,1.5",help="delimited list input")
    
    args = parser.parse_args()

    #---------------------------------------------------------------------------------------------------------------------------------------
    from datasets.nerf_test_poses import SubjectTestPoseLoader
    data_root_fp = "/home/ubuntu/data/"
    render_n_samples = 1024
    target_sample_batch_size = 1 << 20
    grid_resolution = [400, 400, 100]

    #---------------------------------------------------------------------------------------------------------------------------------------
    testPoses = SubjectTestPoseLoader(subject_id=args.scene,root_fp=data_root_fp,numberOfFrames=120, downscale_factor=2)
    testPoses.camtoworlds = testPoses.camtoworlds.to(device)
    testPoses.K = testPoses.K.to(device)

    #---------------------------------------------------------------------------------------------------------------------------------------
    contraction_type = ContractionType.AABB
    args.aabb = [testPoses.aabb[0][0], testPoses.aabb[0][1], -30, testPoses.aabb[1][0], testPoses.aabb[1][1], 30]
    render_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)

    render_step_size = ((render_aabb[3:] - render_aabb[:3]).max() * math.sqrt(3) / render_n_samples).item()
    alpha_thre = 0.0
    occ_thre = 0.1
    print("Using aabb", args.aabb, render_step_size)

    #---------------------------------------------------------------------------------------------------------------------------------------
    # setup the radiance field we want to train.
    step = 0
    max_steps = args.max_steps
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPradianceField(aabb=args.aabb,unbounded=False,).to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],gamma=0.33)
    occupancy_grid = OccupancyGrid(roi_aabb=args.aabb,resolution=grid_resolution,contraction_type=contraction_type).to(device)

    #---------------------------------------------------------------------------------------------------------------------------------------
    # Load checkpoints
    savepath = os.path.join(data_root_fp,args.scene,args.exp_name)
    args.ckpt_path = os.path.join(savepath, "ckpts")
    load_ckpt = sorted(glob.glob(f'{args.ckpt_path}/*.ckpt'))
    if args.ckpt_path != "" and load_ckpt != []: 
        load_ckpt = load_ckpt[-1]
        model = torch.load(load_ckpt)
        step = model['step']+1
        grad_scaler.load_state_dict(model['grad_scaler_state_dict']) # not critical
        radiance_field.load_state_dict(model['radiance_field_state_dict'])
        optimizer.load_state_dict(model['optimizer_state_dict'])
        scheduler.load_state_dict(model['scheduler_state_dict']) # not critical
        occupancy_grid.load_state_dict(model['occupancy_grid_state_dict'])
        print(f"Loaded checkpoint from: {load_ckpt}")
        print(f"Previous Training Loss: loss={model['loss']:.5f}")

    #---------------------------------------------------------------------------------------------------------------------------------------
    NGPGUI(radiance_field, occupancy_grid, render_aabb, testPoses, render_step_size, args.cone_angle, alpha_thre).render()
    dpg.destroy_context()


