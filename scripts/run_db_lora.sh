MODEL_NAME="CompVis/stable-diffusion-v1-4"

# Train IMMA weights agaisnt DreamBooth + LoRA
OUTPUT_DIR="results/dreambooth_lora"
INSTANCE_PROMPT="a photo of *s purse"
INSTANCE_DIR="data/luggage_purse1"

accelerate launch train/defend_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --learning_rate_defense=2e-5 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=1000 \
  --seed="0" \
  --inner_loop_steps=1 \
  --outer_loop_steps=1 \

INSTANCE_PROMPT="A photo of a &m purse"
VALIDATION_PROMPT="A &m purse on the beach"
IMMA_CKPT="${OUTPUT_DIR}/imma_unet_xatten_layer.pt"
accelerate launch train/train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="${INSTANCE_PROMPT}" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --report_to="tensorboard" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="${VALIDATION_PROMPT}" \
  --validation_steps=1 \
  --seed="0" \
  --num_validation_images=4 \
  --imma_ckpt=$IMMA_CKPT