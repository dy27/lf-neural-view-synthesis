import matplotlib.pyplot as plt 
import sys

# experiment_names = [
#     'single',
#     'full',
#     'uniform',
#     'random',
#     # 'dynamic1',
#     # 'dynamic2',
#     # 'dynamic3',
#     # 'dynamic4',
#     'view_corner',
#     'dynamic5',
#     # 'view1',
#     'fixed_random',
#     'fixed_view',
# ]

experiment_names = [
    'dense_single',
    'dense_full',
    'dense_uniform',
    'dense_random',
    # 'dynamic1',
    # 'dynamic2',
    # 'dynamic3',
    # 'dynamic4',
    'dense_view_corner',
    'dense_dynamic',
    # 'view1',
    'dense_fixed_random',
    'dense_fixed_view',
]

eval_paths = [f'/home/david/datasets/seq34/results/{name}/eval.txt' for name in experiment_names]
loss_paths = [f'/home/david/datasets/seq34/results/{name}/loss.txt' for name in experiment_names]

eval_files = [open(path) for path in eval_paths]
loss_files = [open(path) for path in loss_paths]

# Read in data
data = {}
for i_exp, experiment_name in enumerate(experiment_names):
    data[experiment_name] = {
        'iter': [],
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
        data_dict['iter'].append(d[0])
        data_dict['psnr'].append(d[1])
        data_dict['minpsnr'].append(d[2])
        data_dict['maxpsnr'].append(d[3])
        data_dict['ssim'].append(d[4])
    for i, line in enumerate(loss_files[i_exp]):
        l = float(line.strip())
        data_dict['loss_iter'].append(i)
        data_dict['loss'].append(l)


STYLE = 'default'
# STYLE = 'Solarize_Light2'
# STYLE = 'fivethirtyeight'

plt.figure(figsize=(8, 5))
plt.style.use(STYLE)
# plt.plot(data['single']['iter'], data['single']['psnr'])
# plt.plot(data['full']['iter'], data['full']['psnr'])
# plt.plot(data['down16']['iter'], data['down16']['psnr'])
# plt.plot(data['down16_random']['iter'], data['down16_random']['psnr'])
# plt.plot(data['dynamic1']['iter'], data['dynamic1']['psnr'])
for experiment in experiment_names:
    plt.plot(data[experiment]['iter'], data[experiment]['psnr'], linewidth=2)
plt.xlabel('Training Iterations')
plt.ylabel('PSNR (dB)')
plt.legend(['Conventional Imaging', 'Full Light Field', 'Uniform Sampling', 'Random Sampling', 'View-Based Sampling', 'Image Gradient Sampling', 'Fixed Random Sampling', 'Fixed View Sampling'],
    bbox_to_anchor=(1.02, 0.5), loc='center left')

plt.tight_layout()
# plt.legend(experiment_names)
plt.grid()
plt.show()



plt.figure(figsize=(8, 5))
plt.style.use(STYLE)
# plt.plot(data['single']['iter'], data['single']['ssim'])
# plt.plot(data['full']['iter'], data['full']['ssim'])
# plt.plot(data['down16']['iter'], data['down16']['ssim'])
# plt.plot(data['down16_random']['iter'], data['down16_random']['ssim'])
# plt.plot(data['dynamic1']['iter'], data['dynamic1']['ssim'])
for experiment in experiment_names:
    plt.plot(data[experiment]['iter'], data[experiment]['ssim'], linewidth=2)
plt.xlabel('Training Iterations')
plt.ylabel('SSIM')
# plt.legend(['Single View', 'Full Light Field', 'x16 Ray Downsampling', 'x16 Random Ray Downsampling', 'Dynamic1'])
plt.legend(['Conventional Imaging', 'Full Light Field', 'Uniform Sampling', 'Random Sampling', 'View-Based Sampling', 'Image Gradient Sampling',  'Fixed Random Sampling', 'Fixed View Sampling'],
    bbox_to_anchor=(1.02, 0.5), loc='center left')

plt.tight_layout()
# plt.legend(experiment_names)
plt.grid()
plt.show()

# plt.plot(data['single']['loss_iter'], data['single']['loss'])
# plt.plot(data['full']['loss_iter'], data['full']['loss'])
# plt.xlabel('Training Iterations')
# plt.ylabel('Loss')
# plt.legend(['Single View', 'Full Light Field'])
# plt.show()
