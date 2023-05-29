import matplotlib.pyplot as plt 
import sys

dir_path = '/home/david/datasets/seq34/results/FULL_VIEW_TEST'

experiments = [
    'view1',
    'view2',
    'view3',
    'view4',
]

eval_paths = [f'{dir_path}/{name}/eval.txt' for name in experiments]

experiments = ['full'] + experiments
eval_paths = ['/home/david/datasets/seq34/results/full/eval.txt'] + eval_paths

eval_files = [open(path) for path in eval_paths]

# Read in data
data = {}
for i_exp, experiment_name in enumerate(experiments):
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
# for experiment in experiments:
#     plt.plot(data[experiment]['iter'], data[experiment]['psnr'], linewidth=2)


c = ['C6', 'C10', 'C8', 'C9']
c = ['violet', 'mediumslateblue', 'dodgerblue', 'limegreen']

plt.plot(data['full']['iter'], data['full']['psnr'], linewidth=2, color='C1')
plt.plot(data['view1']['iter'], data['view1']['psnr'], linewidth=2, color=c[0])
plt.plot(data['view2']['iter'], data['view2']['psnr'], linewidth=2, color=c[1])
plt.plot(data['view3']['iter'], data['view3']['psnr'], linewidth=2, color=c[2])
plt.plot(data['view4']['iter'], data['view4']['psnr'], linewidth=2, color=c[3])

plt.xlabel('Training Iterations')
plt.ylabel('PSNR (dB)')
# plt.legend(['Conventional Imaging', 'Full Light Field', 'Uniform Sampling', 'Random Sampling', 'View-Based Sampling', 'Image Gradient Sampling'],
#     bbox_to_anchor=(1.02, 0.5), loc='center left')

plt.tight_layout()
plt.legend(experiments)
plt.legend(['Full Light Field', '$d_v=1$', '$d_v=2$', '$d_v=3$', '$d_v=4$'])
plt.grid()
plt.ylim(bottom=20.8)
plt.show()


plt.figure(figsize=(8, 5))
plt.style.use(STYLE)
# plt.plot(data['single']['iter'], data['single']['psnr'])
# plt.plot(data['full']['iter'], data['full']['psnr'])
# plt.plot(data['down16']['iter'], data['down16']['psnr'])
# plt.plot(data['down16_random']['iter'], data['down16_random']['psnr'])
# plt.plot(data['dynamic1']['iter'], data['dynamic1']['psnr'])
# for experiment in experiments:
#     plt.plot(data[experiment]['iter'], data[experiment]['ssim'], linewidth=2)

plt.plot(data['full']['iter'], data['full']['ssim'], linewidth=2, color='C1')
plt.plot(data['view1']['iter'], data['view1']['ssim'], linewidth=2, color=c[0])
plt.plot(data['view2']['iter'], data['view2']['ssim'], linewidth=2, color=c[1])
plt.plot(data['view3']['iter'], data['view3']['ssim'], linewidth=2, color=c[2])
plt.plot(data['view4']['iter'], data['view4']['ssim'], linewidth=2, color=c[3])

plt.xlabel('Training Iterations')
plt.ylabel('SSIM')
# plt.legend(['Conventional Imaging', 'Full Light Field', 'Uniform Sampling', 'Random Sampling', 'View-Based Sampling', 'Image Gradient Sampling'],
#     bbox_to_anchor=(1.02, 0.5), loc='center left')

plt.tight_layout()
plt.legend(experiments)
plt.legend(['Full Light Field', '$d_v=1$', '$d_v=2$', '$d_v=3$', '$d_v=4$'])
plt.grid()
plt.show()