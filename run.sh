#!/usr/bin/env bash
<<com
* Testing performance change of wavelet dim
* model size : 1.15 MB
* wavelet dim : 200, 400, 600, 800, 1000
* My optimal goal is to find best wavelet dim - hidden dim combination for desired model sizes
com
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --width 45 --wavelet_dim 800 --nepoches 20 --first_activation 'swinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --width 105 --wavelet_dim 200 --nepoches 20 --first_activation 'swinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --width 76 --wavelet_dim 400 --nepoches 20 --first_activation 'swinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --width 58 --wavelet_dim 600 --nepoches 20 --first_activation 'swinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --output_file 'dataset1_w64.nc' --all --quantizing --testing
