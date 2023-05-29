import sys
import shutil
import os
from os import listdir
from os.path import isfile, join

N_CAMERAS = 17
extract_camera_index_set = set(i for i in range(N_CAMERAS))  # Set containing the indices of cameras to extract to the new dataset

if __name__ == '__main__':

   original_dataset_folder_path = sys.argv[1]
   new_dataset_folder_path = sys.argv[2]

   for i_camera in range(N_CAMERAS):
      if i_camera in extract_camera_index_set:
         camera_path = f'{original_dataset_folder_path}/{i_camera}'
         image_files = [f for f in listdir(camera_path) if isfile(join(camera_path, f))]
         image_files.sort()

         for i_image, image_name in enumerate(image_files):
            print(i_camera, i_image, image_name)

            # Only extract every 5th frame between frames 52 and 142
            if i_image < 52:
               continue
            if i_image == 142:
               break
            if i_image % 5 != 2:
               continue
            i_image = (i_image-52 )// 5


            _, img_ext = os.path.splitext(image_name)
            image_path = f'{camera_path}/{image_name}'
            new_image_name = f'{i_camera}_{i_image}'
            new_image_path = f'{new_dataset_folder_path}/images/{new_image_name}{img_ext}'
            # print(new_image_path)
            shutil.copy2(image_path, new_image_path)
      





   