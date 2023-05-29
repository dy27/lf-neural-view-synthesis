import matplotlib.pyplot as plt 
import sys
import numpy as np

img_paths = [f'/home/david/datasets/seq34/images/{i}_10.png' for i in range(17)]
mask_paths = [f'/home/david/datasets/seq34/images/dynamic_mask_{i}_10.png' for i in range(17)]

images = [plt.imread(path) for path in img_paths]
masks = [plt.imread(path) for path in mask_paths]

# masked_imgs = [images[i] * masks[i] for i in range(17)]

# images = masked_imgs


images = masks

# for i in range(17):
#     img = images[i]
#     mask = masks[i]
#     h, w, _ = img.shape
#     for y in range(h):
#         for x in range(w):
#             # print(mask[y,x,:])
#             if np.array_equal(mask[y,x,:], np.array([1.,1.,1.])):
#                 # print('hi')
#                 img[y,x,:] = 1


cam_map = [
    [-1, -1, -1, -1,  0, -1, -1, -1, -1],
    [-1, -1, -1, -1,  1, -1, -1, -1, -1],
    [-1, -1, -1, -1,  2, -1, -1, -1, -1],
    [-1, -1, -1, -1,  3, -1, -1, -1, -1],
    [ 4,  5,  6,  7,  8,  9, 10, 11, 12],
    [-1, -1, -1, -1, 13, -1, -1, -1, -1],
    [-1, -1, -1, -1, 14, -1, -1, -1, -1],
    [-1, -1, -1, -1, 15, -1, -1, -1, -1],
    [-1, -1, -1, -1, 16, -1, -1, -1, -1],
]
n_rows = len(cam_map)
n_cols = len(cam_map[0])


f, axarr = plt.subplots(9,9,figsize=(8,6))


for i in range(n_rows):
    for j in range(n_cols):
        axarr[i,j].axis('off')
        image_index = cam_map[i][j]
        if image_index == -1:
            continue
        axarr[i,j].imshow(images[image_index])
        

# axarr[0,0].imshow(image_datas[0])
# axarr[0,1].imshow(image_datas[1])
# axarr[1,0].imshow(image_datas[2])
# axarr[1,1].imshow(image_datas[3])

f.tight_layout()

plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=1,
                    wspace=0.0,
                    hspace=0.03)

# plt.subplots_adjust(wspace=0.0,
#                     hspace=0.05)

plt.savefig("view1.png", format="png", dpi=200)

# plt.subplot_tool()
plt.show()