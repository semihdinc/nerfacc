"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import argparse
import math
import os
import time
import glob

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from radiance_fields.ngp import NGPradianceField
from utils import render_image, set_random_seed
from torch.utils.tensorboard import SummaryWriter

from nerfacc import ContractionType, OccupancyGrid

if __name__ == "__main__":

    device = "cuda:0"
    set_random_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_split",type=str,default="train",choices=["train", "trainval", "None"],help="which train split to use")
    parser.add_argument("--root_dir",type=str,default="/home/ubuntu/ws/data/",help="Root directory of the scenes")
    parser.add_argument("--scene",type=str,default="lego",help="which scene to use")
    parser.add_argument("--aabb",type=lambda s: [float(item) for item in s.split(",")],default="-1.5,-1.5,-1.5,1.5,1.5,1.5",help="delimited list input")
    parser.add_argument("--test_chunk_size",type=int,default=8192)
    parser.add_argument("--ev_data",action="store_true",help="whether to use EV dataset or not")
    parser.add_argument("--unbounded",action="store_true",help="whether to use unbounded rendering")
    parser.add_argument("--auto_aabb",action="store_true",help="whether to automatically compute the aabb")
    parser.add_argument("--cone_angle", type=float, default=0.0)
    parser.add_argument("--i_test",type=int,default=5000,help="Iterations to start validation test")
    parser.add_argument("--i_ckpt",type=int,default=1000,help="Iterations to save model")
    parser.add_argument("--ckpt_path",type=str,default="",help="Model ckpt path to save/load")
    parser.add_argument("--render_n_samples",type=int,default=1024,help="Number of samples per render")
    parser.add_argument("--max_steps",type=int,default=30000,help="Number of iterations")
    args = parser.parse_args()

    render_n_samples = args.render_n_samples

    # setup the dataset
    train_dataset_kwargs = {}
    test_dataset_kwargs = {}

    if args.ev_data == True:
        from datasets.nerf_synthetic import SubjectLoader
        from datasets.generateTestPoses import SubjectTestPoseLoader
        data_root_fp = args.root_dir
        train_dataset_kwargs = {"color_bkgd_aug": "random"}
        target_sample_batch_size = 1 << 20
        grid_resolution = 256

    elif args.unbounded:
        from datasets.nerf_360_v2 import SubjectLoader
        data_root_fp = args.root_dir
        target_sample_batch_size = 1 << 20
        train_dataset_kwargs = {"color_bkgd_aug": "random", "factor": 4}
        test_dataset_kwargs = {"factor": 4}
        grid_resolution = 256

    else:
        from datasets.nerf_synthetic import SubjectLoader
        data_root_fp = args.root_dir
        target_sample_batch_size = 1 << 18
        grid_resolution = 128

    train_dataset = SubjectLoader(subject_id=args.scene,root_fp=data_root_fp,split=args.train_split,
                                  num_rays=target_sample_batch_size // render_n_samples,**train_dataset_kwargs)
    train_dataset.images = train_dataset.images.to(device)
    train_dataset.camtoworlds = train_dataset.camtoworlds.to(device)
    train_dataset.K = train_dataset.K.to(device)

    test_dataset = SubjectLoader(subject_id=args.scene,root_fp=data_root_fp,split="None",
                                 num_rays=None,**test_dataset_kwargs)
    test_dataset.images = test_dataset.images.to(device)
    test_dataset.camtoworlds = test_dataset.camtoworlds.to(device)
    test_dataset.K = test_dataset.K.to(device)

    #test poses
    numOfFrames = 30
    test_poses = SubjectTestPoseLoader(subject_id=args.scene,root_fp=data_root_fp, numberOfFrames=numOfFrames, **train_dataset_kwargs)
    test_poses.camtoworlds = test_poses.camtoworlds.to(device)
    test_poses.K = test_poses.K.to(device)

    savepath = os.path.join(data_root_fp,args.scene+"_test")
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        print('Test results folder not found, creating new dir: ' + savepath)
    else:
        print('Test images will be saved in ' +savepath)

    if args.auto_aabb:
        #camera_locs = torch.cat([train_dataset.camtoworlds, test_dataset.camtoworlds])[:, :3, -1]
        camera_locs = train_dataset.camtoworlds[:, :3, -1]
        args.aabb = torch.cat([camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]).tolist()
        print("Using auto aabb", args.aabb)

    # setup the scene bounding box.
    if args.unbounded:
        print("Using unbounded rendering")
        contraction_type = ContractionType.UN_BOUNDED_SPHERE
        # contraction_type = ContractionType.UN_BOUNDED_TANH
        scene_aabb = None
        near_plane = 0.2
        far_plane = 1e4
        render_step_size = 1e-2
        alpha_thre = 1e-2
    else:
        contraction_type = ContractionType.AABB
        # args.aabb = [-1.5, -1.5, -1.5, 1.5, 1.5, 1.5]
        args.aabb = [-1500, -1500, -50, 1500, 1500, 250]
        scene_aabb = torch.tensor(args.aabb, dtype=torch.float32, device=device)
        near_plane = None
        far_plane = None
        render_step_size = ((scene_aabb[3:] - scene_aabb[:3]).max() * math.sqrt(3) / render_n_samples).item()
        alpha_thre = 0.0

    print("Using aabb", args.aabb, render_step_size)

    # setup the radiance field we want to train.
    max_steps = args.max_steps
    grad_scaler = torch.cuda.amp.GradScaler(2**10)
    radiance_field = NGPradianceField(aabb=args.aabb,unbounded=args.unbounded,).to(device)
    optimizer = torch.optim.Adam(radiance_field.parameters(), lr=1e-2, eps=1e-15)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[max_steps // 2, max_steps * 3 // 4, max_steps * 9 // 10],
        gamma=0.33,
    )

    occupancy_grid = OccupancyGrid(roi_aabb=args.aabb,resolution=grid_resolution,contraction_type=contraction_type).to(device)
    writer = SummaryWriter(savepath)
    print(f'Tensorboard cmd: tensorboard --logdir {savepath}')

    # training
    step = 0
    if args.ckpt_path != "": 
        load_ckpt = sorted(glob.glob(f'{args.ckpt_path}/*.ckpt'))[-1]
        torch.load(load_ckpt)
        print(f'Loaded checkpoint from: {load_ckpt}')

    tic = time.time()
    for epoch in range(10000000):
        for i in range(len(train_dataset)):
            radiance_field.train()
            data = train_dataset[i]

            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            def occ_eval_fn(x):
                if args.cone_angle > 0.0:
                    # randomly sample a camera for computing step size.
                    camera_ids = torch.randint(0, len(train_dataset), (x.shape[0],), device=device)
                    origins = train_dataset.camtoworlds[camera_ids, :3, -1]
                    t = (origins - x).norm(dim=-1, keepdim=True)

                    # compute actual step size used in marching, based on the distance to the camera.
                    step_size = torch.clamp(t * args.cone_angle, min=render_step_size)
                    
                    # filter out the points that are not in the near far plane.
                    if (near_plane is not None) and (far_plane is not None):
                        step_size = torch.where(
                            (t > near_plane) & (t < far_plane),
                            step_size,
                            torch.zeros_like(step_size),
                        )
                else:
                    step_size = render_step_size
                
                # compute occupancy
                density = radiance_field.query_density(x)
                
                return density * step_size

            # update occupancy grid
            occupancy_grid.every_n_step(step=step, occ_eval_fn=occ_eval_fn)

            # render
            rgb, acc, depth, n_rendering_samples = render_image(
                radiance_field,
                occupancy_grid,
                rays,
                scene_aabb,
                # rendering options
                near_plane=near_plane,
                far_plane=far_plane,
                render_step_size=render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=args.cone_angle,
                alpha_thre=alpha_thre,
            )
            if n_rendering_samples == 0:
                continue

            # dynamic batch size for rays to keep sample batch size constant.
            num_rays = len(pixels)
            num_rays = int(num_rays * (target_sample_batch_size / float(n_rendering_samples)))

            train_dataset.update_num_rays(num_rays)
            alive_ray_mask = acc.squeeze(-1) > 0

            # compute loss
            loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

            optimizer.zero_grad()
            # do not unscale it because we are using Adam.
            grad_scaler.scale(loss).backward()
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                elapsed_time = time.time() - tic
                mse = F.mse_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])
                psnr = -10.0 * torch.log(mse) / np.log(10.0)
                writer.add_scalar("mse/train", mse, step)
                writer.add_scalar("psnr/train", psnr, step)
                print(
                    f"elapsed_time={elapsed_time:.2f}s | step={step} | "
                    f"loss={mse:.5f} ({psnr:.2f})| "
                    f"alive_ray={alive_ray_mask.long().sum():d} | "
                    f"n_renders={n_rendering_samples:d} | n_rays={len(pixels):d} |"
                )

            if step >= 0 and step % args.i_test == 0 and step > 0:
                # evaluation
                radiance_field.eval()

                psnrs, mses = [], []
                with torch.no_grad():
                    for i in tqdm.tqdm(range(10)):
                        data = test_dataset[i]
                        # data = test_poses[i]
                        render_bkgd = data["color_bkgd"]
                        rays = data["rays"]
                        pixels = data["pixels"]

                        # rendering
                        rgb, acc, depth, _ = render_image(
                            radiance_field,
                            occupancy_grid,
                            rays,
                            scene_aabb,
                            # rendering options
                            near_plane=near_plane,
                            far_plane=far_plane,
                            render_step_size=render_step_size,
                            render_bkgd=render_bkgd,
                            cone_angle=args.cone_angle,
                            alpha_thre=alpha_thre,
                            # test options
                            test_chunk_size=args.test_chunk_size,
                        )
                        mse = F.mse_loss(rgb, pixels)
                        mses.append(mse)
                        psnr = -10.0 * torch.log(mse) / np.log(10.0)
                        psnrs.append(psnr.item())
                        writer.add_image(f'rgb{i}', rgb, step, dataformats='HWC')
                        depth -= depth.min()
                        depth /= depth.max()
                        writer.add_image(f'depth{i}', depth, step, dataformats='HWC')
                        
                        # saveImg = os.path.join(savepath,"rgb_test_"+str(i)+".png")
                        # imageio.imwrite(saveImg,(rgb.cpu().numpy() * 255).astype(np.uint8))
                        # imageio.imwrite("/home/ubuntu/data/depth_test_"+str(i)+".png",(depth.cpu().numpy() * 255).astype(np.uint8))
                # Write to tensorboard
                mse_avg = sum(mses) / len(mses)
                psnr_avg = sum(psnrs) / len(psnrs)
                print(f"evaluation: mse_avg={mse_avg:.5f} | psnr_avg={psnr_avg:.3f}")
                writer.add_scalar("psnr/val", psnr_avg, step)
                writer.add_scalar("mse/val", mse_avg, step)
                train_dataset.training = True

            if step >= 0 and step % args.i_ckpt == 0 and step > 0:
                # Save checkpoint
                ckpt_flag = True # Save flag
                args.ckpt_path = os.path.join(savepath, "ckpts") if args.ckpt_path == "" else args.ckpt_path
                os.makedirs(args.ckpt_path, exist_ok=True)
                ckpt_path = os.path.join(args.ckpt_path, f'model_{step}.ckpt')
                for ckpt in sorted(glob.glob(f'{args.ckpt_path}/*.ckpt')):
                    if int(os.path.basename(ckpt)[6:-5]) <= step or ckpt == []:
                        os.remove(ckpt)
                    else:
                        print(f'Higher checkpoint is found at: {ckpt}')
                        print('Skip saving checkpoint')
                        ckpt_flag = False

                if ckpt_flag:
                    torch.save({
                                'step': step,
                                'grad_scaler_state_dict': grad_scaler.state_dict(),
                                'radiance_field_state_dict': radiance_field.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'occupancy_grid_state_dict': occupancy_grid.state_dict(),
                                'loss': loss
                                }, ckpt_path)
                    print(f'Checkpoint save in: {ckpt_path}')

            if step == max_steps:
                # End of training
                print("training stops")
                writer.flush()
                exit()

            step += 1
