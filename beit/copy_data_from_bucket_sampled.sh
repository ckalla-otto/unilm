#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")"; pwd -P)"
DATA_FOLDER="generated/cache/hackathon"
mkdir -p $DATA_FOLDER
gsutil cp gs://brain-ai-image-dev-artifacts/hackathon_2022-02/tensorflow_datasets/dataset_builder_hackathon/0.2.0/dataset_builder_hackathon-train.tfrecord-00019-of-00256 $DATA_FOLDER/dataset_builder_hackathon-train.tfrecord-00019-of-00256
gsutil cp gs://brain-ai-image-dev-artifacts/hackathon_2022-02/tensorflow_datasets/dataset_builder_hackathon/0.2.0/dataset_builder_hackathon-test.tfrecord-00000-of-00256 $DATA_FOLDER/dataset_builder_hackathon-test.tfrecord-00000-of-00256

