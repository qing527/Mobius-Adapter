# matterport3d
CUDA_VISIBLE_DEVICES=0 python eval.py --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
    --checkpoint   /data/liuqing/MobiusAdapter/train_outputs/stage2/controlnet\
    --input_rgb_dir /data/liuqing/Dataset/MP3D_unifuse/ \
    --test_list_file  /data/liuqing/MobiusAdapter/dataset/matterport3d_test.txt  \
    --output_dir /data/liuqing/MobiusAdapter/results/mp3d \
    --color_map jet \
    --denoise_steps 10 \
    --batch_size 2

# stanford2d3d
CUDA_VISIBLE_DEVICES=0 python eval.py --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
    --checkpoint   /data/liuqing/MobiusAdapter/train_outputs/stage2/controlnet\
    --input_rgb_dir /data/liuqing/Dataset/Stanford2d3d/ \
    --test_list_file  /data/liuqing/MobiusAdapter/dataset/stanford2d3d_test.txt  \
    --output_dir /data/liuqing/MobiusAdapter/results/sf2d3d \
    --test_name 'stanford2d3d' \
    --color_map jet \
    --denoise_steps 10 \
    --batch_size 2

# pano3d
OPENCV_IO_ENABLE_OPENEXR=1 CUDA_VISIBLE_DEVICES=4 python eval.py --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
    --checkpoint /data/liuqing/MobiusAdapter/train_outputs/stage2/controlnet\
    --test_name pano3d \
    --pano3d_root /data/liuqing/Dataset/Pano3D/ \
    --pano3d_part gibson_v2 \
    --pano3d_split /data/liuqing/MobiusAdapter/dataset/gv2_test.txt \
    --output_dir /data/liuqing/MobiusAdapter/results/pano3d \
    --color_map jet \
    --denoise_steps 10 \
    --batch_size 2

# kitti360
CUDA_VISIBLE_DEVICES=5 python eval.py --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
    --checkpoint /data/liuqing/MobiusAdapter/train_outputs/stage2/controlnet \
    --test_name kitti360 \
    --output_dir /data/liuqing/MobiusAdapter/results/kitti360 \
    --color_map jet \
    --denoise_steps 10 \
    --batch_size 2
