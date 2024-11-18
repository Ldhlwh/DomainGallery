# MODEL_NAME="CompVis/stable-diffusion-v1-4"
MODEL_NAME="../diffusion_models/stable-diffusion-v1-4_diffusers"

SOURCE_NOUN="face"
IDENTIFIER="sks"

accelerate launch --num_processes 1 --gpu_ids 0 train_domaingallery.py --seed 0 \
  --mode "prior_attribute_erasure" --output_dir "runs/pre_erased/${IDENTIFIER}_${SOURCE_NOUN}" \
  --pretrained_model_name_or_path $MODEL_NAME  \
  --source_data_dir "source_image/${SOURCE_NOUN}" --source_prompt "a ${SOURCE_NOUN}" --num_source_images 1000 \
  --max_train_steps 500 --train_batch_size 4 --learning_rate 1e-4  --lr_warmup_steps 0 \
  --start_checkpointing_steps 0 --checkpointing_steps 100 \
  --start_validation_steps 0 --validation_steps 100 \
  --gradient_checkpointing --use_8bit_adam \
  --offset_noise_scale 0.1 
