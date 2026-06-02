<h1>Mobius Adapter: Boosting Zero-Shot 360◦ Depth Estimation in  Diffusion Models via Mobius Convolution</h1>

## Datasets

Please download the preferred datasets,  i.e., [PNVS](https://github.com/bluestyle97/PNVS), [Matterport3D](https://niessner.github.io/Matterport/), [Stanford2D3D](http://3dsemantics.stanford.edu/), and [Kitti360](https://www.cvlibs.net/datasets/kitti-360/). For Matterport3D and Stanford2D3D, please preprocess them following [UniFuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion).


## Pre-trained Models
You can download checkpoints from [link](https://huggingface.co/qing527/MobiusAdapter/tree/main) and place them under `train_outputs/stage1/unet` and `train_outputs/stage2/controlnet`.


## Usage

### Prepraration

```bash
git clone https://github.com/qing527/MobiusAdapter
conda create -n mobius python=3.10
conda activate mobius
cd MobiusAdapter
pip install -r requirements.txt
```

**Note**: We use python==3.10, and pytorch==2.1.1, cuda==12.1, and cudnn==8.9.2.
# Training 

### stage1
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/config_float32.yaml train_stage1.py \
                  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
                  --dataset_name "PNVS" \
                  --output_dir "train_outputs/stage1" \
                  --train_batch_size 1 \
                  --max_train_steps 18000 \
                  --gradient_accumulation_steps 16 \
                  --learning_rate 3e-5 \
                  --random_flip \
                  --lr_warmup_steps 0 \
                  --dataloader_num_workers 0 \
                  --tracker_project_name "PNVS" \
                  --noise_type "pyramid_noise" \
                  --gradient_checkpointing \
                  --enable_xformers_memory_efficient_attention
```

### stage2
```
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/config_float32.yaml train_stage2.py \
                  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
                  --checkpoint train_outputs/stage1 \
                  --dataset_name "PNVS" \
                  --output_dir "train_outputs/stage2" \
                  --train_batch_size 1 \
                  --max_train_steps 10000 \
                  --gradient_accumulation_steps 2 \
                  --gradient_checkpointing \
                  --learning_rate 1e-5 \
                  --random_flip \
                  --lr_warmup_steps 0 \
                  --dataloader_num_workers 0 \
                  --validation_steps 1000 \
                  --checkpointing_steps 1000 \
                  --tracker_project_name "pnvs_mobiusadapter" \
                  --enable_xformers_memory_efficient_attention
```
The training script is in `scripts/train.sh`.

# Evaluation  

```
CUDA_VISIBLE_DEVICES=0 python eval.py --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
    --checkpoint  $CHECKPOINT_DIR\
    --input_rgb_dir $DATASET_ROOT_DIR \
    --test_list_file  $TEST_LIST_DIR  \
    --output_dir $OUTPUT_DIR \
    --color_map jet \
    --denoise_steps 10 \
    --batch_size 2


```
The eval script is in `scripts/eval.sh`.


## Acknowledgement

We thank the authors of the projects below:  
*[Marigold](https://github.com/prs-eth/Marigold)*, *[diffuser](https://github.com/huggingface/diffusers)*, *[MobiusConv](https://github.com/twmitchel/MobiusConv)*, *[ControlNet](https://github.com/lllyasviel/ControlNet)*.
