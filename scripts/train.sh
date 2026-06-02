# stage1
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/config_float32.yaml train_stage1.py \
                  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
                  --dataset_name "PNVS" \
                  --output_dir "train_outputs" \
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

# stage2
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file config/config_float32.yaml train_stage2.py \
                  --pretrained_model_name_or_path 'stabilityai/stable-diffusion-2-1' \
                  --checkpoint /data/liuqing/MobiusAdapter/train_outputs/stage1 \
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