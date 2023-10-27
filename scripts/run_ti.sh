MODEL_NAME="CompVis/stable-diffusion-v1-4"

OUTPUT_DIR="results/textual_inversion"
TOKEN='<l*p>'
INIT='purse'
INSTANCE_PROMPT="a photo of ${TOKEN}"
INSTANCE_DIR="data/luggage_purse1"

accelerate launch train/defend_textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$INSTANCE_DIR \
    --learnable_property=object \
    --placeholder_token="${TOKEN}" --initializer_token="${INIT}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=1000 \
    --learning_rate=5e-04 --scale_lr \
    --learning_rate_defense=1e-06 \
    --lr_scheduler="constant" \
    --report_to="tensorboard" \
    --lr_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --seed="0" \
    --inner_loop_steps=1 \
    --outer_loop_steps=1 


NEW_TOKEN="<m&q>"
VALIDATION_PROMPT="A photo of ${NEW_TOKEN} on the beach"
IMMA_CKPT="${OUTPUT_DIR}/imma_unet_xatten_layer.pt"
accelerate launch train/train_textual_inversion.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --train_data_dir=$INSTANCE_DIR \
    --learnable_property=object \
    --placeholder_token="${TOKEN}" --initializer_token="${INIT}" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=2000 \
    --learning_rate=5.0e-04 --scale_lr \
    --lr_scheduler="constant" \
    --report_to="tensorboard" \
    --lr_warmup_steps=0 \
    --output_dir=$OUTPUT_DIR \
    --validation_prompt="${VALIDATION_PROMPT}" \
    --validation_epochs=4 \
    --num_validation_images=4 \
    --seed="0" \
    --imma_ckpt=$IMMA_CKPT