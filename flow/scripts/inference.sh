# :)
SEED=051197

CKPT_PATH="checkpoint.pt"
BATCH_SIZE=2
VAE_FRAME_DECODE_BATCH=4

python -m flow.dnn.sample --inference-idx demo --ckpt-path $CKPT_PATH \
 --dnn-spec FlowField_S/4 \
 --temporal-res 72 \
 --batch-size $BATCH_SIZE \
 --vae-frame-decode-batch $VAE_FRAME_DECODE_BATCH \
 --solver heun \
 --sample-fps 24 \
 --learnable-pe \
 --use-temporal-attention \
 --seed  $SEED
