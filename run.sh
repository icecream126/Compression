CUDA_VISIBLE_DEVICES=0 python train.py --nepoches 1 --first_activation 'gelu' --activation 'gelu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=1 python train.py --nepoches 1 --first_activation 'relu' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=2 python train.py --nepoches 1 --first_activation 'siren' --activation 'siren' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=3 python train.py --nepoches 1 --first_activation 'swinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=4 python train.py --nepoches 1 --first_activation 'wire' --activation 'wire' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --output_file 'dataset1_w64.nc' --all --quantizing --testing
CUDA_VISIBLE_DEVICES=5 python train.py --nepoches 1 --first_activation 'shinr' --activation 'relu' --variable 'z' --dataloader_mode 'sampling_nc' --file_name 'dataset1.nc' --width 64 --max_order 7 --output_file 'dataset1_w64.nc' --all --quantizing --testing
