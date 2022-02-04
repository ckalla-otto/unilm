#!/bin/bash
OUTPUT_DIR="output_dir"
DEST_DIR="gs://brain-ai-image-dev-artifacts/dali_usecase-image-similarity/training/beit_orig_finetuning"
while true  
do  
  gsutil rsync -r $OUTPUT_DIR $DEST_DIR
  sleep 300  
done