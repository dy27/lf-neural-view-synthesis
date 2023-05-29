#!/usr/bin/env python3

# Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from fileinput import filename
from functools import total_ordering
import png
import sys
import os
from os import listdir
from os.path import isfile, join
from natsort import natsorted
import numpy as np
import cv2

WRITE_IMG = False

# view_weight_map = {
#     8: 0,

#     3: 1,
#     7: 1,
#     9: 1,
#     13: 1,

#     2: 2,
#     6: 2,
#     10: 2,
#     14: 2,

#     1: 3,
#     5: 3,
#     11: 3,
#     15: 3,

#     0: 4,
#     4: 4,
#     12: 4,
#     16: 4,
# }

view_weight_map = {
    8: 0,

    3: 0,
    7: 0,
    9: 0,
    13: 0,

    2: 0,
    6: 0,
    10: 0,
    14: 0,

    1: 0,
    5: 0,
    11: 0,
    15: 0,

    0: 1,
    4: 1,
    12: 1,
    16: 1,
}

# view_weight_map = {
#     8: 0,

#     3: 1,
#     7: 1,
#     9: 1,
#     13: 1,

#     2: 0,
#     6: 0,
#     10: 0,
#     14: 0,

#     1: 0,
#     5: 0,
#     11: 0,
#     15: 0,

#     0: 0,
#     4: 0,
#     12: 0,
#     16: 0,
# }

def redistribute_weights(view_weight_map, sample_rate):
    n_views = len(view_weight_map)
    max_weight = max(view_weight_map.values())
    for k in view_weight_map:
        view_weight_map[k] /= max_weight

    weights = []
    for i in range(n_views):
        weights.append(view_weight_map[i])
    max_weight = max(weights)
    weights = [w / max_weight for w in weights]
    total_weights = sum(weights)

    inv_weights = [1 - w for w in weights]
    total_inv_weights = sum(inv_weights)

    target_sum = n_views * sample_rate

    if total_weights > target_sum:
        return

    inverse_scale_factor = (target_sum - total_weights) / total_inv_weights
    extra_add = [inverse_scale_factor * w for w in inv_weights]

    new_weights = [weights[i] + extra_add[i] for i in range(n_views)]

    for i in range(n_views):
        view_weight_map[i] = new_weights[i]

    # return view_weight_map


sampling_map = None

fixed_cache = {}

CUT_BOTTOM = False

def generate_training_mask(width, height, image_path, mode='random', sample_rate=1, i_camera=None, i_frame=None, fixed=False):

    split = image_path.split('/')
    dir_path = '/'.join(split[:-1])
    file_name = split[-1]

    mask_file_name = 'dynamic_mask_' + file_name
    mask_file_path = dir_path + '/' + mask_file_name
    print('Creating mask at ' + mask_file_path)

    
    global WRITE_IMG
    if i_camera == 8 and i_frame == 10:
        # WRITE_IMG = True
        pass

    if fixed and i_camera in fixed_cache:
        boolean_mask = fixed_cache[i_camera]
    elif mode == 'dynamic':
        boolean_mask = generate_mask_dynamic(width, height, image_path, sample_rate)
    elif mode == 'view':
        boolean_mask = generate_random_boolean_mask(width, height, sampling_map[i_camera])
    elif mode == 'uniform':
        boolean_mask = generate_mask_uniform(width, height, sample_rate)
    else:
        boolean_mask = generate_random_boolean_mask(width, height, sample_rate)

    if fixed and i_camera not in fixed_cache:
        fixed_cache[i_camera] = boolean_mask

    WRITE_IMG = False

    mask_img = []
    for y in range(height):
        row = []
        for x in range(width):
            if boolean_mask[y][x] == 0:
                row.extend((0, 0, 0))
            else:
                row.extend((255, 255, 255))
        mask_img.append(row)

    with open(mask_file_path, 'wb') as f:
        w = png.Writer(width, height, greyscale=False)
        w.write(f, mask_img)



def generate_random_boolean_mask(width, height, sample_rate):
    n_pixels = width * height
    n_selected_pixels = int(sample_rate * n_pixels)
    boolean_array = np.concatenate((np.zeros(n_selected_pixels), np.ones(n_pixels-n_selected_pixels)))
    np.random.shuffle(boolean_array)
    boolean_mask = np.reshape(boolean_array, (height, width))
    return boolean_mask


def generate_mask_uniform(width, height, sample_rate):
    pixel_gap = int(1/sample_rate)
    n_pixels = width * height
    boolean_array = np.ones(n_pixels)
    for i in range(n_pixels):
        if i % pixel_gap == 0:
            boolean_array[i] = 0
    boolean_mask = np.reshape(boolean_array, (height, width))
    return boolean_mask


# def generate_mask_uniform(width, height, sample_rate):
#     pixel_gap = int(1/sample_rate)

#     root_num = int(pixel_gap**0.5)

#     boolean_mask = np.ones((height,width))
#     for i in range(height):
#         for j in range(width):
#             if i % root_num == 0 and j % root_num == 0:
#                 boolean_mask[i][j] = 0
#     return boolean_mask


def generate_mask_dynamic(width, height, image_path, sample_rate=1):

    if CUT_BOTTOM:
        height -= 12

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(3,3),cv2.BORDER_DEFAULT)
    gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
    gray = cv2.GaussianBlur(gray,(7,7),cv2.BORDER_DEFAULT)

    gX = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=1, dy=0)
    gY = cv2.Sobel(gray, ddepth=cv2.CV_64F, dx=0, dy=1)
    mag = np.sqrt((gX ** 2) + (gY ** 2))
    # orientation = np.arctan2(gY, gX) * (180 / np.pi) % 180

    if WRITE_IMG:
        cv2.imwrite('tmp_gradients.png', mag)

    if CUT_BOTTOM:
        mag = mag[:-12,:]


    n_pixels_target = width * height * sample_rate

    mag += 0.1 * np.max(mag)
    X = mag / np.max(mag)
    X_sum = np.sum(X)

    gain = X_sum / n_pixels_target
    if gain > 1:
        X /= gain
        X_sum /= gain
 
    X_inv = 1 - X
    X_inv_sum = np.sum(X_inv)

    inverse_scale_factor = (n_pixels_target - X_sum) / X_inv_sum
    extra_add = inverse_scale_factor * X_inv 

    prob = X + extra_add

    if WRITE_IMG:
        cv2.imwrite('tmp_prob.png', prob*255)

    # cv2.imshow('prob', prob)
    # cv2.waitKey(0)

    ref = np.random.uniform(size=(height,width))

    boolean_mask = prob < ref

    if CUT_BOTTOM:
        height += 12
        new_boolean_mask = np.ones((height,width))
        new_boolean_mask[:-12,:] = boolean_mask
        boolean_mask = new_boolean_mask

    return boolean_mask


# Assumes all views have same number of rays
def view_weight_to_sample_map(view_weight_map, sample_rate):
    print(view_weight_map)
    redistribute_weights(view_weight_map, sample_rate)
    print('redistributed')
    print(view_weight_map)
    total_weight = sum(view_weight_map.values())
    n_cameras = len(view_weight_map)

    for i_camera, weight in view_weight_map.items():
        normalised_weight =  weight / total_weight
    
    sample_map = {}
    for i_camera, weight in view_weight_map.items():
        normalised_weight =  weight / total_weight  # Value in range [0,1], represents the percentage of total subsampled rays which should be in this camera subview
        camera_sample_rate = n_cameras * sample_rate * normalised_weight
        assert camera_sample_rate <= 1 + 1e-11, camera_sample_rate
        sample_map[i_camera] = camera_sample_rate
    
    # print(sample_rate, n_cameras, sum(sample_map.values()))
    # print(n_cameras * sample_rate)
    # print(sample_map)
    assert abs(sum(sample_map.values()) - n_cameras * sample_rate) < 1e-11

    return sample_map


def generate_training_masks(width, height, dir_path, mode='random', sample_rate=1, fixed=False):

    if mode == 'view':
        global sampling_map
        sampling_map = view_weight_to_sample_map(view_weight_map, sample_rate)

    image_files = [f for f in listdir(dir_path) if isfile(join(dir_path, f))]
    image_files = natsorted(image_files)

    for file in image_files:
        if 'dynamic_mask_' in file:
            continue

        full_path = dir_path + '/' + file
        i_camera, tmp = file.split('_')
        i_camera = int(i_camera)
        i_frame = int(tmp.split('.')[0])
        # mode = 1 if i_camera == 8 else 0
        # mode = 1

        generate_training_mask(width, height, full_path, mode, sample_rate, i_camera, i_frame, fixed)






import argparse
import os
import commentjson as json

import numpy as np

import shutil
import time

from common import *
from scenes import *

from tqdm import tqdm

import pyngp as ngp # noqa

import matplotlib.pyplot as plt

EXPERIMENT_PATH = '/home/david/datasets/seq34/results/data_performance'

DATA_PATH = '/home/david/datasets/seq34/images'

# sampling_rates = [n/16 for n in range(1,17)]
# sampling_rates = [n/8/16 for n in range(1,8)]
# for n in range(1,17):
#     sampling_rates.append(n/16)
# sampling_rates = [1/16]

sampling_rates = [n/8/16 for n in range(1,8,2)] + [n/16 for n in range(1,17,2)]
# sampling_rates = sampling_rates[6:]
# sampling_rates = [1/2, 1/3, 1/4, 1/8, 1/16, 1/32, 1/64]

# sampling_rates = [1]

print('sampling_rates:', sampling_rates)


SAVE_FREQ = 100
log_file = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run neural graphics primitives testbed with additional configuration & output options")

    parser.add_argument("--scene", "--training_data", default="", help="The scene to load. Can be the scene's name or a full path to the training data.")
    parser.add_argument("--mode", default="", const="nerf", nargs="?", choices=["nerf", "sdf", "image", "volume"], help="Mode can be 'nerf', 'sdf', 'image' or 'volume'. Inferred from the scene if unspecified.")
    parser.add_argument("--network", default="", help="Path to the network config. Uses the scene's default if unspecified.")

    parser.add_argument("--load_snapshot", default="", help="Load this snapshot before training. recommended extension: .msgpack")
    parser.add_argument("--save_snapshot", default="", help="Save this snapshot after training. recommended extension: .msgpack")

    parser.add_argument("--nerf_compatibility", action="store_true", help="Matches parameters with original NeRF. Can cause slowness and worse results on some scenes.")
    parser.add_argument("--test_transforms", default="", help="Path to a nerf style transforms json from which we will compute PSNR.")
    parser.add_argument("--near_distance", default=-1, type=float, help="Set the distance from the camera at which training rays start for nerf. <0 means use ngp default")
    parser.add_argument("--exposure", default=0.0, type=float, help="Controls the brightness of the image. Positive numbers increase brightness, negative numbers decrease it.")

    parser.add_argument("--screenshot_transforms", default="", help="Path to a nerf style transforms.json from which to save screenshots.")
    parser.add_argument("--screenshot_frames", nargs="*", help="Which frame(s) to take screenshots of.")
    parser.add_argument("--screenshot_dir", default="", help="Which directory to output screenshots to.")
    parser.add_argument("--screenshot_spp", type=int, default=16, help="Number of samples per pixel in screenshots.")

    parser.add_argument("--video_camera_path", default="", help="The camera path to render, e.g., base_cam.json.")
    parser.add_argument("--video_camera_smoothing", action="store_true", help="Applies additional smoothing to the camera trajectory with the caveat that the endpoint of the camera path may not be reached.")
    parser.add_argument("--video_fps", type=int, default=60, help="Number of frames per second.")
    parser.add_argument("--video_n_seconds", type=int, default=1, help="Number of seconds the rendered video should be long.")
    parser.add_argument("--video_spp", type=int, default=8, help="Number of samples per pixel. A larger number means less noise, but slower rendering.")
    parser.add_argument("--video_output", type=str, default="video.mp4", help="Filename of the output video.")

    parser.add_argument("--save_mesh", default="", help="Output a marching-cubes based mesh from the NeRF or SDF model. Supports OBJ and PLY format.")
    parser.add_argument("--marching_cubes_res", default=256, type=int, help="Sets the resolution for the marching cubes grid.")

    parser.add_argument("--width", "--screenshot_w", type=int, default=0, help="Resolution width of GUI and screenshots.")
    parser.add_argument("--height", "--screenshot_h", type=int, default=0, help="Resolution height of GUI and screenshots.")

    parser.add_argument("--gui", action="store_true", help="Run the testbed GUI interactively.")
    parser.add_argument("--train", action="store_true", help="If the GUI is enabled, controls whether training starts immediately.")
    parser.add_argument("--n_steps", type=int, default=-1, help="Number of steps to train for before quitting.")
    parser.add_argument("--second_window", action="store_true", help="Open a second window containing a copy of the main output.")

    parser.add_argument("--sharpen", default=0, help="Set amount of sharpening applied to NeRF training images. Range 0.0 to 1.0.")


    args = parser.parse_args()
    return args


def evaluate_model(args, testbed):
    print("Evaluating test transforms from ", args.test_transforms)
    with open(args.test_transforms) as f:
        test_transforms = json.load(f)
    data_dir=os.path.dirname(args.test_transforms)
    totmse = 0
    totpsnr = 0
    totssim = 0
    totcount = 0
    minpsnr = 1000
    maxpsnr = 0

    # Evaluate metrics on black background
    testbed.background_color = [0.0, 0.0, 0.0, 1.0]

    # Prior nerf papers don't typically do multi-sample anti aliasing.
    # So snap all pixels to the pixel centers.
    testbed.snap_to_pixel_centers = True
    spp = 8

    testbed.nerf.rendering_min_transmittance = 1e-4

    testbed.fov_axis = 0
    testbed.fov = test_transforms["camera_angle_x"] * 180 / np.pi
    testbed.shall_train = False

    with tqdm(list(enumerate(test_transforms["frames"])), unit="images", desc=f"Rendering test frame") as t:
        for i, frame in t:
            p = frame["file_path"]
            if "." not in p:
                p = p + ".png"
            ref_fname = os.path.join(data_dir, p)
            if not os.path.isfile(ref_fname):
                ref_fname = os.path.join(data_dir, p + ".png")
                if not os.path.isfile(ref_fname):
                    ref_fname = os.path.join(data_dir, p + ".jpg")
                    if not os.path.isfile(ref_fname):
                        ref_fname = os.path.join(data_dir, p + ".jpeg")
                        if not os.path.isfile(ref_fname):
                            ref_fname = os.path.join(data_dir, p + ".exr")

            ref_image = read_image(ref_fname)

            # NeRF blends with background colors in sRGB space, rather than first
            # transforming to linear space, blending there, and then converting back.
            # (See e.g. the PNG spec for more information on how the `alpha` channel
            # is always a linear quantity.)
            # The following lines of code reproduce NeRF's behavior (if enabled in
            # testbed) in order to make the numbers comparable.
            if testbed.color_space == ngp.ColorSpace.SRGB and ref_image.shape[2] == 4:
                # Since sRGB conversion is non-linear, alpha must be factored out of it
                ref_image[...,:3] = np.divide(ref_image[...,:3], ref_image[...,3:4], out=np.zeros_like(ref_image[...,:3]), where=ref_image[...,3:4] != 0)
                ref_image[...,:3] = linear_to_srgb(ref_image[...,:3])
                ref_image[...,:3] *= ref_image[...,3:4]
                ref_image += (1.0 - ref_image[...,3:4]) * testbed.background_color
                ref_image[...,:3] = srgb_to_linear(ref_image[...,:3])

            # if i == 0:
            write_image(f"run_imgs/{i}_ref.png", ref_image)

            testbed.set_nerf_camera_matrix(np.matrix(frame["transform_matrix"])[:-1,:])
            image = testbed.render(ref_image.shape[1], ref_image.shape[0], spp, True)

            # if i == 0:
            write_image(f"run_imgs/{i}_out.png", image)

            # Original Code
            # diffimg = np.absolute(image - ref_image)
            # diffimg[...,3:4] = 1.0

            # Modified Code
            image = image[...,:3]
            diffimg = np.absolute(image - ref_image)

            # if i == 0:
            write_image(f"run_imgs/{i}_diff.png", diffimg)

            A = np.clip(linear_to_srgb(image[...,:3]), 0.0, 1.0)
            R = np.clip(linear_to_srgb(ref_image[...,:3]), 0.0, 1.0)
            mse = float(compute_error("MSE", A, R))
            ssim = float(compute_error("SSIM", A, R))
            totssim += ssim
            totmse += mse
            psnr = mse2psnr(mse)
            totpsnr += psnr
            minpsnr = psnr if psnr<minpsnr else minpsnr
            maxpsnr = psnr if psnr>maxpsnr else maxpsnr
            totcount = totcount+1
            t.set_postfix(psnr = totpsnr/(totcount or 1))

    psnr_avgmse = mse2psnr(totmse/(totcount or 1))
    psnr = totpsnr/(totcount or 1)
    ssim = totssim/(totcount or 1)
    print(f"PSNR={psnr} [min={minpsnr} max={maxpsnr}] SSIM={ssim}")
    return (psnr, minpsnr, maxpsnr, ssim)


def init_testbed(args):
    mode = ngp.TestbedMode.Nerf
    configs_dir = os.path.join(ROOT_DIR, "configs", "nerf")
    scenes = scenes_nerf

    base_network = os.path.join(configs_dir, "base.json")
    if args.scene in scenes:
        network = scenes[args.scene]["network"] if "network" in scenes[args.scene] else "base"
        base_network = os.path.join(configs_dir, network+".json")
    network = args.network if args.network else base_network
    if not os.path.isabs(network):
        network = os.path.join(configs_dir, network)

    testbed = ngp.Testbed(mode)
    testbed.nerf.sharpen = float(args.sharpen)
    testbed.exposure = args.exposure
    if mode == ngp.TestbedMode.Sdf:
        testbed.tonemap_curve = ngp.TonemapCurve.ACES

    if args.scene:
        scene = args.scene
        if not os.path.exists(args.scene) and args.scene in scenes:
            scene = os.path.join(scenes[args.scene]["data_dir"], scenes[args.scene]["dataset"])
        testbed.load_training_data(scene)


    if args.load_snapshot:
        snapshot = args.load_snapshot
        if not os.path.exists(snapshot) and snapshot in scenes:
            snapshot = default_snapshot_filename(scenes[snapshot])
        print("Loading snapshot ", snapshot)
        testbed.load_snapshot(snapshot)
    else:
        testbed.reload_network_from_file(network)

    testbed.shall_train = True

    testbed.nerf.render_with_camera_distortion = True

    network_stem = os.path.splitext(os.path.basename(network))[0]
    if args.mode == "sdf":
        setup_colored_sdf(testbed, args.scene)

    if args.near_distance >= 0.0:
        print("NeRF training ray near_distance ", args.near_distance)
        testbed.nerf.training.near_distance = args.near_distance

    if args.nerf_compatibility:
        print(f"NeRF compatibility mode enabled")

        # Prior nerf papers accumulate/blend in the sRGB
        # color space. This messes not only with background
        # alpha, but also with DOF effects and the likes.
        # We support this behavior, but we only enable it
        # for the case of synthetic nerf data where we need
        # to compare PSNR numbers to results of prior work.
        testbed.color_space = ngp.ColorSpace.SRGB

        # No exponential cone tracing. Slightly increases
        # quality at the cost of speed. This is done by
        # default on scenes with AABB 1 (like the synthetic
        # ones), but not on larger scenes. So force the
        # setting here.
        testbed.nerf.cone_angle_constant = 0

        # Optionally match nerf paper behaviour and train on a
        # fixed white bg. We prefer training on random BG colors.
        # testbed.background_color = [1.0, 1.0, 1.0, 1.0]
        # testbed.nerf.training.random_bg_color = False

    return testbed



def train_testbed(testbed, n_steps):
    old_training_step = 0

    # If we loaded a snapshot, didn't specify a number of steps, _and_ didn't open a GUI,
    # don't train by default and instead assume that the goal is to render screenshots,
    # compute PSNR, or render a video.
    if n_steps < 0:
        n_steps = 2000

    # save_path = f'run_models/0.msgpack'
    # print("Saving snapshot ", save_path)
    # testbed.save_snapshot(save_path, False)

    losses = []

    tqdm_last_update = 0
    if n_steps > 0:
        with tqdm(desc="Training", total=n_steps, unit="step") as t:
            while testbed.frame():

                # if testbed.training_step % SAVE_FREQ == 0:
                # 	save_path = f'run_models/{testbed.training_step}.msgpack'
                # 	print("Saving snapshot ", save_path)
                # 	testbed.save_snapshot(save_path, False)
                # 	# psnr, minpsnr, maxpsnr, ssim = evaluate_model(args, testbed)
                # 	# log_file.write(f'{testbed.training_step},{testbed.loss},{psnr},{minpsnr},{maxpsnr},{ssim}')

                losses.append(testbed.loss)
                
                if testbed.want_repl():
                    repl(testbed)
                # What will happen when training is done?
                if testbed.training_step >= n_steps:
                    break

                # Update progress bar
                if testbed.training_step < old_training_step or old_training_step == 0:
                    old_training_step = 0
                    t.reset()

                now = time.monotonic()
                if now - tqdm_last_update > 0.1:
                    t.update(testbed.training_step - old_training_step)
                    t.set_postfix(loss=testbed.loss)
                    old_training_step = testbed.training_step
                    tqdm_last_update = now

    return losses



if __name__ == "__main__":
    # if log_file is None:
    # 	log_file =  open('run_log.txt', 'w')


    # log_file.close()

    args = parse_args()

    args.mode = args.mode or mode_from_scene(args.scene) or mode_from_scene(args.load_snapshot)
    if not args.mode:
        raise ValueError("Must specify either a valid '--mode' or '--scene' argument.")


    eval_results = []

    for sr in sampling_rates:

        fixed_cache.clear()

        generate_training_masks(256, 192, DATA_PATH, 'view', sr, fixed=True)

        testbed = init_testbed(args)

        losses = train_testbed(testbed, args.n_steps)

        if args.test_transforms:
            psnr, minpsnr, maxpsnr, ssim = evaluate_model(args, testbed)
            eval_results.append((psnr, minpsnr, maxpsnr, ssim))
            with open('tmp_eval_results.txt', 'a') as f:
                f.write(f'{sr},{psnr},{minpsnr},{maxpsnr},{ssim}\n')
                    
        if args.save_snapshot:
            print("Saving snapshot ", args.save_snapshot)
            testbed.save_snapshot(args.save_snapshot, False)

        # with open('loss.txt', 'w') as f:
        # 	for n in losses:
        # 		f.write(str(n) + '\n')


    with open('eval_results.txt', 'w') as f:
        for i, sr in enumerate(sampling_rates):
            psnr, minpsnr, maxpsnr, ssim = eval_results[i]
            f.write(f'{sr},{psnr},{minpsnr},{maxpsnr},{ssim}\n')


    psnrs = [x[0] for x in eval_results]
    ssims = [x[3] for x in eval_results]
    plt.figure()
    plt.plot(sampling_rates, psnrs)
    plt.show()

    plt.figure()
    plt.plot(sampling_rates, ssims)
    plt.show()



    

