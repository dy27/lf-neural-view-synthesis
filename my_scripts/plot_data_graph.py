import matplotlib.pyplot as plt 
import sys

def sample_rate_to_ray_count(sample_rate):
    return sample_rate * 256 * 192 * 17 * 2

experiment_names = [
    'single',
    'full',
    'uniform',
    'random',
    'view',
    'dynamic_subsample',
    'fixed_random',
    'fixed_view',
]

experiment_names += [
    'dense_single',
    'dense_full',
    'dense_uniform',
    'dense_random',
    # 'dynamic1',
    # 'dynamic2',
    # 'dynamic3',
    # 'dynamic4',
    'dense_view',
    'dense_dynamic',
    # 'view1',
    'dense_fixed_random',
    'dense_fixed_view',
]

eval_paths = [f'/home/david/datasets/seq34/results/data_performance/{name}/eval_data.txt' for name in experiment_names]

eval_files = [open(path) for path in eval_paths]

# Read in data
data = {}
for i_exp, experiment_name in enumerate(experiment_names):
    data[experiment_name] = {
        'sample_rate': [],
        'psnr': [],
        'minpsnr': [],
        'maxpsnr': [],
        'ssim': [],
        'loss_iter': [],
        'loss': []
    }
    data_dict = data[experiment_name]
    for line in eval_files[i_exp]:
        d = line.split(',')
        d = list(map(lambda x: float(x), d))
        data_dict['sample_rate'].append(d[0])
        data_dict['psnr'].append(d[1])
        data_dict['minpsnr'].append(d[2])
        data_dict['maxpsnr'].append(d[3])
        data_dict['ssim'].append(d[4])



# plt.figure(figsize=(12, 8))
# for experiment in experiment_names:
#     plt.plot(data[experiment]['sample_rate'], data[experiment]['psnr'], linestyle='--', marker='o', markersize=7)
# plt.xlabel('Sample Rate')
# plt.ylabel('PSNR (dB)')
# # plt.legend(['Single View', 'Full Light Field', 'x16 Ray Downsampling', 'x16 Random Ray Downsampling', 'Dynamic1'])
# plt.legend(experiment_names)
# plt.show()


# plt.figure(figsize=(8, 6))
# ray_counts = [sample_rate_to_ray_count(sr) for sr in data['single']['sample_rate']]
# plt.plot(ray_counts, data['single']['ssim'], linestyle='--', marker='o', markersize=6, color='C0')
# ray_counts = [sample_rate_to_ray_count(sr) for sr in data['full']['sample_rate']]
# plt.plot(ray_counts, data['full']['ssim'], linestyle='--', marker='o', markersize=6, color='C2')

# plt.xlabel('Number of Rays Trained On')
# plt.ylabel('SSIM')
# plt.legend(['Conventional Image', 'Full Light Field Image'])
# # plt.legend(experiment_names)
# plt.show()

LEGEND = ['Conventional Imaging', 'Full Light Field', 'Uniform Sampling', 'Random Sampling', 'View-Based Sampling', 'Image Gradient Sampling', 'Fixed Random Sampling', 'Fixed View-Based Sampling']


plt.figure(figsize=(8, 5))
# plt.tight_layout()
# for experiment in experiment_names:
#     ray_counts = [sample_rate_to_ray_count(sr) for sr in data[experiment]['sample_rate']]
#     if len(ray_counts) == 1:
#         plt.scatter(ray_counts, data[experiment]['ssim'], marker='D', s=81)
#     else:
#         plt.plot(ray_counts, data[experiment]['ssim'], linestyle='--', marker='o', markersize=6)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['single']['sample_rate']]
plt.scatter(ray_counts, data['single']['psnr'], marker='D', s=81, color='C0', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['full']['sample_rate']]
plt.scatter(ray_counts, data['full']['psnr'], marker='D', s=81, color='C1', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['uniform']['sample_rate']]
plt.plot(ray_counts, data['uniform']['psnr'], linestyle='--', marker='o', markersize=6, color='C2')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['random']['sample_rate']]
plt.plot(ray_counts, data['random']['psnr'], linestyle='--', marker='o', markersize=6, color='C3')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['view']['sample_rate']]
plt.plot(ray_counts, data['view']['psnr'], linestyle='--', marker='o', markersize=6, color='C4')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dynamic_subsample']['sample_rate']]
plt.plot(ray_counts, data['dynamic_subsample']['psnr'], linestyle='--', marker='o', markersize=6, color='C5')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['fixed_random']['sample_rate']]
plt.plot(ray_counts, data['fixed_random']['psnr'], linestyle='--', marker='o', markersize=6, color='C6')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['fixed_view']['sample_rate']]
plt.plot(ray_counts, data['fixed_view']['psnr'], linestyle='--', marker='o', markersize=6, color='C7')
plt.xlabel('Number of Rays')
plt.ylabel('PSNR (dB)')
plt.legend(['Conventional Image', 'Random Light Field Sampling', 'Full Light Field Image'])
plt.legend(experiment_names)
plt.legend(LEGEND)
plt.grid()
plt.show()


plt.figure(figsize=(8, 5))
# plt.tight_layout()
# for experiment in experiment_names:
#     ray_counts = [sample_rate_to_ray_count(sr) for sr in data[experiment]['sample_rate']]
#     if len(ray_counts) == 1:
#         plt.scatter(ray_counts, data[experiment]['ssim'], marker='D', s=81)
#     else:
#         plt.plot(ray_counts, data[experiment]['ssim'], linestyle='--', marker='o', markersize=6)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['single']['sample_rate']]
plt.scatter(ray_counts, data['single']['ssim'], marker='D', s=81, color='C0', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['full']['sample_rate']]
plt.scatter(ray_counts, data['full']['ssim'], marker='D', s=81, color='C1', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['uniform']['sample_rate']]
plt.plot(ray_counts, data['uniform']['ssim'], linestyle='--', marker='o', markersize=6, color='C2')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['random']['sample_rate']]
plt.plot(ray_counts, data['random']['ssim'], linestyle='--', marker='o', markersize=6, color='C3')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['view']['sample_rate']]
plt.plot(ray_counts, data['view']['ssim'], linestyle='--', marker='o', markersize=6, color='C4')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dynamic_subsample']['sample_rate']]
plt.plot(ray_counts, data['dynamic_subsample']['ssim'], linestyle='--', marker='o', markersize=6, color='C5')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['fixed_random']['sample_rate']]
plt.plot(ray_counts, data['fixed_random']['ssim'], linestyle='--', marker='o', markersize=6, color='C6')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['fixed_view']['sample_rate']]
plt.plot(ray_counts, data['fixed_view']['ssim'], linestyle='--', marker='o', markersize=6, color='C7')
plt.xlabel('Number of Rays')
plt.ylabel('SSIM')
plt.legend(['Conventional Image', 'Random Light Field Sampling', 'Full Light Field Image'])
plt.legend(experiment_names)
plt.legend(LEGEND)
plt.grid()
plt.show()



# plt.figure(figsize=(6, 4))
# for experiment in experiment_names:
#     ray_counts = [sample_rate_to_ray_count(sr) for sr in data[experiment]['sample_rate']]
#     plt.plot(ray_counts, data[experiment]['ssim'], linestyle='--', marker='o', markersize=6)
# plt.xlabel('Number of Rays')
# plt.ylabel('Structural Similarity (SSIM)')
# plt.legend(['Conventional Image', 'Random Light Field Sampling'])
# plt.legend(experiment_names)
# plt.xlim(left=10000, right=130000)
# # plt.ylim([0.5, 0.7])
# plt.ylim(top=0.65)
# plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.show()









plt.figure(figsize=(8, 5))
# plt.tight_layout()
# for experiment in experiment_names:
#     ray_counts = [sample_rate_to_ray_count(sr) for sr in data[experiment]['sample_rate']]
#     if len(ray_counts) == 1:
#         plt.scatter(ray_counts, data[experiment]['ssim'], marker='D', s=81)
#     else:
#         plt.plot(ray_counts, data[experiment]['ssim'], linestyle='--', marker='o', markersize=6)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_single']['sample_rate']]
plt.scatter(ray_counts, data['dense_single']['psnr'], marker='D', s=81, color='C0', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_full']['sample_rate']]
plt.scatter(ray_counts, data['dense_full']['psnr'], marker='D', s=81, color='C1', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_uniform']['sample_rate']]
plt.plot(ray_counts, data['dense_uniform']['psnr'], linestyle='--', marker='o', markersize=6, color='C2')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_random']['sample_rate']]
plt.plot(ray_counts, data['dense_random']['psnr'], linestyle='--', marker='o', markersize=6, color='C3')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_view']['sample_rate']]
plt.plot(ray_counts, data['dense_view']['psnr'], linestyle='--', marker='o', markersize=6, color='C4')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_dynamic']['sample_rate']]
plt.plot(ray_counts, data['dense_dynamic']['psnr'], linestyle='--', marker='o', markersize=6, color='C5')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_fixed_random']['sample_rate']]
plt.plot(ray_counts, data['dense_fixed_random']['psnr'], linestyle='--', marker='o', markersize=6, color='C6')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_fixed_view']['sample_rate']]
plt.plot(ray_counts, data['dense_fixed_view']['psnr'], linestyle='--', marker='o', markersize=6, color='C7')
plt.xlabel('Number of Rays')
plt.ylabel('PSNR (dB)')
plt.legend(['Conventional Image', 'Random Light Field Sampling', 'Full Light Field Image'])
plt.legend(experiment_names)
plt.legend(LEGEND)
plt.grid()
plt.show()


plt.figure(figsize=(8, 5))
# plt.tight_layout()
# for experiment in experiment_names:
#     ray_counts = [sample_rate_to_ray_count(sr) for sr in data[experiment]['sample_rate']]
#     if len(ray_counts) == 1:
#         plt.scatter(ray_counts, data[experiment]['ssim'], marker='D', s=81)
#     else:
#         plt.plot(ray_counts, data[experiment]['ssim'], linestyle='--', marker='o', markersize=6)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_single']['sample_rate']]
plt.scatter(ray_counts, data['dense_single']['ssim'], marker='D', s=81, color='C0', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_full']['sample_rate']]
plt.scatter(ray_counts, data['dense_full']['ssim'], marker='D', s=81, color='C1', zorder=100)
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_uniform']['sample_rate']]
plt.plot(ray_counts, data['dense_uniform']['ssim'], linestyle='--', marker='o', markersize=6, color='C2')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_random']['sample_rate']]
plt.plot(ray_counts, data['dense_random']['ssim'], linestyle='--', marker='o', markersize=6, color='C3')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_view']['sample_rate']]
plt.plot(ray_counts, data['dense_view']['ssim'], linestyle='--', marker='o', markersize=6, color='C4')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_dynamic']['sample_rate']]
plt.plot(ray_counts, data['dense_dynamic']['ssim'], linestyle='--', marker='o', markersize=6, color='C5')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_fixed_random']['sample_rate']]
plt.plot(ray_counts, data['dense_fixed_random']['ssim'], linestyle='--', marker='o', markersize=6, color='C6')
ray_counts = [sample_rate_to_ray_count(sr) for sr in data['dense_fixed_view']['sample_rate']]
plt.plot(ray_counts, data['dense_fixed_view']['ssim'], linestyle='--', marker='o', markersize=6, color='C7')
plt.xlabel('Number of Rays')
plt.ylabel('SSIM')
plt.legend(['Conventional Image', 'Random Light Field Sampling', 'Full Light Field Image'])
plt.legend(experiment_names)
plt.legend(LEGEND)
plt.grid()
plt.show()
