import sys
import json

N_CAMERAS = 17
use_cameras = set(i for i in range(N_CAMERAS))
# use_cameras = set([8])

"""
Returns 0 if the specified frame should be assigned to the training set, 1 for the test set, and -1 for neither.
"""
def assign_frame(frame: dict) -> int:
    
    image_path = frame['file_path']
    print(image_path)
    image_name = image_path.split('/')[-1]
    name, ext = image_name.split('.')

    i_camera, i_frame = name.split('_')
    i_camera = int(i_camera)
    i_frame = int(i_frame)

    # if i_camera not in use_cameras:
    #     return -1
    # return 0
    # if 10 <= i_frame <= 14:
    #     return 1

    frames = {0, 2, 4, 6, 14, 17}
    if i_frame in frames:
        return 0
    return -1

    if i_frame == 0 or i_frame == 17:
        return 0
    else:
        return 1

    return 0


if __name__ == '__main__':

    transforms_path = sys.argv[1]

    split = transforms_path.split('/')
    transforms_dir_path = '/'.join(split[:-1])
    transforms_file_name = split[-1]

    data = json.load(open(transforms_path))

    train_data = json.loads(json.dumps(data))
    train_data['frames'] = []
    test_data = json.loads(json.dumps(train_data))

    for frame in data['frames']:
        assigned_set = assign_frame(frame)
        if assigned_set == 0:
            train_data['frames'].append(frame)
        elif assigned_set == 1:
            test_data['frames'].append(frame)
        else:
            print('Skipping frame')

    name, ext = transforms_file_name.split('.')
    train_file_path = transforms_dir_path + '/' + name + '_train543.' + ext
    test_file_path = transforms_dir_path + '/' + name + '_test543.' + ext
    
    if len(train_data['frames']) > 0:
        with open(train_file_path, 'w') as f:
            json.dump(train_data, f, indent=4)

    if len(test_data['frames']) > 0:
        with open(test_file_path, 'w') as f:
            json.dump(test_data, f, indent=4)
