#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
ROOT_DIR=${SCRIPT_DIR}

JOB_NAME=$1
WORKDIR=/beit-embedding-training
LOG_FREQUENCY=180
OUTPUT_LOCATION=gs://brain-ai-image-dev-artifacts/dali_usecase-image-similarity/training/jobs/$JOB_NAME

sync_logs () {\
  while :; do
    echo "Uploading to: $OUTPUT_LOCATION"
    gsutil -m rsync -r ${WORKDIR}/output_dir ${OUTPUT_LOCATION}/
  

    # tensorboard logs?
#    EVENTS=$(compgen -G "lightning_logs/version_0/events.*" | head -n1)
#    if [ -n "$EVENTS" ]; then
#      for filename in lightning_logs/version_0/events.*; do
#        gsutil cp $filename ${OUTPUT_FOLDER}/$filename
#      done
#    fi
    sleep $LOG_FREQUENCY
  done
}

# shellcheck disable=SC2164
cd "${ROOT_DIR}"

#src/data/copy_data_from_bucket_sampled.sh
echo "Copying training data..."
./copy_data_from_bucket_sampled.sh

echo "Starting training"
sync_logs &
SYNC_PID=$!


python run_class_finetuning.py --model beit_base_patch16_224 --data_path "generated/cache/hackathon" \
                --nb_classes 987 --data_set "tfrecord" \
                --finetune "https://unilm.blob.core.windows.net/beit/beit_base_patch16_224_pt22k_ft22k.pth" \
                --output_dir "output_dir" --batch_size 64 --lr 2e-3 \
                --warmup_epochs 5 --epochs 90 --layer_decay 0.75 --drop_path 0.2 \
                --weight_decay 0.05 --layer_scale_init_value 1e-5 --clip_grad 1.0 \
                --device cuda \
                --train_dataset_sample_size 20000 \
                --eval_steps 100 \
                --num_training_steps_per_epoch 1000

echo "Training done!"
# Wait a couple of minutes after training and kill files sync
sleep 400
kill $SYNC_PID
#echo "Copying training artifacts to bucket"
#gsutil cp /sherlock/training.log ${OUTPUT_FOLDER}/training-${TIMESTAMP}-finished.log
#gsutil cp -r /sherlock/models ${OUTPUT_FOLDER}/models
echo "-----    Done Training and Copying Job!   --------"


