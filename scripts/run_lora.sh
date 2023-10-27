MODEL_NAME="CompVis/stable-diffusion-v1-4"

OUTPUT_DIR="results/relearning"
TRAIN_DATA_DIR="data/vangogh"
DELTA_CKPT="diffusers-VanGogh-ESDx1-UNET.pt"

accelerate launch train/defend_text_to_image_lora.py \
  --mixed_precision="no" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --caption_column="prompt" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=50 \
  --learning_rate_lora=1e-04 \
  --learning_rate=1e-05 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --delta_ckpt=$DELTA_CKPT \
  --report_to="tensorboard" \
  --max_train_samples 20 \
  --inner_loop_steps=1 \
  --outer_loop_steps=1 


IMMA_CKPT="${OUTPUT_DIR}/imma_unet_xatten_layer.pt"
VALIDATION_PROMPT="An artwork by Van Gogh"
accelerate launch train/train_text_to_image_lora.py \
  --mixed_precision="fp16" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --caption_column="prompt" \
  --resolution=512 --random_flip \
  --train_batch_size=1 \
  --num_train_epochs=50 \
  --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --validation_prompt="${VALIDATION_PROMPT}" --report_to="tensorboard" \
  --validation_epochs 1 \
  --max_train_samples 20 \
  --delta_ckpt=$DELTA_CKPT \
  --imma_ckpt=$IMMA_CKPT 