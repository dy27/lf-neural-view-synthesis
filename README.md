# Neural Field View Synthesis with Light Field Cameras

This repository provides scripts which interface with NVIDIA's InstantNGP code (https://github.com/NVlabs/instant-ngp).

## Setup

### Install InstantNGP
Install InstantNGP according to the documentation found at https://github.com/NVlabs/instant-ngp. Copy the directories/files in this repository into the base directory of the InstantNGP project.


### Dataset
This project uses an image dataset taken by the EPIModule light field camera, which contains a trajectory of light field images.
The root directory of the dataset is split into one folder for each of the subviews in the light field image, with each folder containing the sequence of frames of the trajectory in sequence. An example is shown in the below directory structure:

```
dataset
└───0
│   │   0000000000.png
│   │   0000000001.png
│   │   ...
│   │   0000000093.png
│   
└───1
│   │   0000000000.png
│   │   0000000001.png
│   │   ...
│   │   0000000093.png
│   
...
|
└───16
    │   0000000000.png
    │   0000000001.png
    │   ...
    │   0000000093.png

```

This dataset is transformed to a different format, where all light field images reside in the same directory, where each image is named `{camera_index}_{frame_number}.png`. This transformation is done by the `epi_dataset_extract_images.py` script. Currently, this script only extracts a specific subset of the frames in the EPIModule dataset. This can be modified in the code to extract a different subset or the full dataset for training.


### Transform files
For training with InstantNGP, a transform .json file is required, which stores the filepaths of every image to be used for training, as well as the corresponding camera transform to each image. Refer to the InstantNGP documentation for more info about this file. This transform file can be generated using the `colmap2nerf.py` script provided in the InstantNGP repository. This script can be setup to perform the entire COLMAP matching process before exporting to the transform file. This is described in the InstantNGP documentation. I had issues getting this to work, and ran COLMAP separately using a standalone COLMAP installation on the dataset directory. I then saved the COLMAP results as a text format, before providing this text file to the `colmap2nerf.py`. This is demonstrated in the `colmap2transforms.sh` shell script.

Examples of different transform .json files are provided in the `data` directory of this repository. This transform files correspond to different sets of input training data used to collect and evaluate results in this project, as well as split training and testing data subsets.

The `train_test_split.py` can be used to split a transform file into different subsets for training and testing. This file takes as input a single transform .json file, and outputs multiple transform files as specified in the code.


### Training masks for pixel sampling
InstantNGP provides functionality to mask out specific pixels to prevent training on particular rays measured by the image. This is done by adding a new image to the dataset directory with 'dynamic_mask_' prepended to the image name (e.g. For `0_0.png`, add a new mask image called `dynamic_mask_0_0.png`). All pixels in the mask image should be either black or white, with black representing pixels selected for training, and white representing pixels masks, and not used for training. Refer to the InstantNGP documentation for more detail.

The `generate_training_masks.py` script takes in the dataset directory, and generates the mask images for training. Different pixel sampling methods are available (uniform, dynamic, random, view-based etc.), and can be selected by modifying the code. The motivation and functionality of these sampling methods are described in the thesis paper associated with this work.



### Training and evaluating the NeRF model
The InstantNGP repository has a `run.py` Python wrapper script which provides a convenient interface to run InstantNGP, conduct experiments, and collect results. This repository contains the scripts `custom_run.py` and `custom_run2.py`, which are modifications from the original `run.py` script which facilitate the relevant experiments used in this thesis work. `custom_run.py` is almost identical to `run.py`, but saves the state of the NeRF model at a specified frequency during the training process (current code saves every 100 training iterations). This is saved in a directory called `run_models` in the main directory of the InstantNGP, which must be created manually by the user. The `custom_train.sh` shell script in this repository shows an example of how `custom_run.py` can be called to train a model.

The NeRF model saved at n iterations during training can then be used to plot all the performance metrics of the model across the training process. The `custom_evaluate.py` script uses the NeRF models storbest quant trading firms rankinged in the `run_models` folder from training, and performs an evaluation of them on a testing dataset, specified by a transform .json file. This should be a different transform file to the one used for training, which contains no data overlap. This script outputs an `eval.txt` file in the current directory. Each line in this text file has the format `{model_iter},{psnr},{minpsnr},{maxpsnr},{ssim}`. For every model in the `run_models` folder, a separate line is written containing the number of iterations the model was trained, the PSNR (average across testing set), minimum and maximum PSNRs across the testing set, and the SSIM score. The `custom_evaluate.sh` shell script shows an example of how the `custom_evaluate.py` script can be called.

`custom_run2.py` evaluates the performance of NeRF models trained using pixel sampling methods with different sampling rates. It runs the original `run.py` in a loop, iterating through a specified list of sampling rates. The `custom_run2.py` contains a copy of the code in `generate_training_masks.py`, which is called every loop to generate a new set of training masks with the sampling rate for that loop. The selected sampling method can also be changed in the code.At the end of every loop, evaluation is also run on the NeRF model; a separate evaluation script is not used for this experiment. The output of this script is a text file called `eval_results.txt`, where each line is in the format: `{sampling_rate},{psnr},{minpsnr},{maxpsnr},{ssim}`. The `custom_train2.sh` shell script shows an example of how the `custom_run2.py` script can be called.



### Plotting results
Various plotting scripts are included in this repository to visualise the output text files of the experiments. More detail for these scripts are provided below.



## Script Summary

`epi_dataset_extract_images.py`: Transforms the EPIModule dataset format into the format used in these experiments, as outlined above.

`generate_training_masks.py`: Operates on an image directory, and creates a mask image for each image in the directory according to the parameters provided.

`plot_data_graph.py`: Plots the data fidelity graph from the thesis work. This requires multiple `eval_results.txt` files, one for each of the sampling methods. Hence, `custom_run2.py` must be run once for every sampling method to collect the data to generate this plot.

`plot_eval.py`: Plots the results contained in an `eval.txt` results file generated from the `custom_run.py` and `custom_evaluate.py` training and evaluation process. Shows the convergence of the model during training for each of the performance metrics.

`plot_lf_view.py`: Used to the generate the figure of an example light field image for the thesis work.

`plot_view_graph.py`: Used to generate the figure from the thesis work for comparing training convergence for different view-based sampling methods, where views at specified distances from the centre view are sampled for training. Requires an `eval.txt` file for every view-based sampling method investigated. Hence, `custom_run.py` and `custom_evaluate.py` must be run once for each method to collect the data for this plot.

`train_test_split.py`: Takes a transform .json file, and splits it into training and testing distributions, outputting new transform files for each.

`custom_evaluate.py`: Uses the `run_models` folder, and runs an evaluation on each saved NeRF model in the folder. Outputs an `eval.txt` text file containing a line for each model's results.

`custom_run.py`: Modified version of `run.py` from the original InstantNGP repository, which saves models into the `run_models` folder.

`custom_run2.py`: Modified version of `run.py` from the original InstantNGP repository, which runs repeated loops of training mask generation, model training, and model evaluation across a range of different sampling rates.

