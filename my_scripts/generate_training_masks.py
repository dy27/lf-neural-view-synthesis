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

# view_weight_map = {
#     8: 0,

#     3: 0,
#     7: 0,
#     9: 0,
#     13: 0,

#     2: 0,
#     6: 0,
#     10: 0,
#     14: 0,

#     1: 0,
#     5: 0,
#     11: 0,
#     15: 0,

#     0: 1,
#     4: 1,
#     12: 1,
#     16: 1,
# }

view_weight_map = {
    8: 0,

    3: 1,
    7: 1,
    9: 1,
    13: 1,

    2: 0,
    6: 0,
    10: 0,
    14: 0,

    1: 0,
    5: 0,
    11: 0,
    15: 0,

    0: 0,
    4: 0,
    12: 0,
    16: 0,
}

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


# def generate_mask_uniform(width, height, sample_rate):
#     pixel_gap = int(1/sample_rate)
#     n_pixels = width * height
#     boolean_array = np.ones(n_pixels)
#     for i in range(n_pixels):
#         if i % pixel_gap == 0:
#             boolean_array[i] = 0
#     boolean_mask = np.reshape(boolean_array, (height, width))
#     return boolean_mask


def generate_mask_uniform(width, height, sample_rate):
    pixel_gap = int(1/sample_rate)

    root_num = int(pixel_gap**0.5)

    boolean_mask = np.ones((height,width))
    for i in range(height):
        for j in range(width):
            if i % root_num == 0 and j % root_num == 0:
                boolean_mask[i][j] = 0
    return boolean_mask


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




if __name__ == '__main__':
    width = int(sys.argv[1])
    height = int(sys.argv[2])
    dir_path = sys.argv[3]
    generate_training_masks(width, height, dir_path, mode='dynamic', sample_rate=1/4, fixed=False)