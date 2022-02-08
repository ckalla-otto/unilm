#!/bin/bash

PROJECT_ID=brain-ai-image-dev
IMAGE_REPO_NAME=dali_image_embedding_training
# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=beit_finetuning_orig_impl_hackathon_data_$(date -u +"%Y%m%d_%H%M%S")
IMAGE_TAG=${JOB_NAME}
REGION=europe-west1

# IMAGE_URI: the complete URI location for Cloud Container Registry
IMAGE_URI=eu.gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG}


echo "-----------------------------"
echo "Building docker image locally"
echo "-----------------------------"
# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

echo "------------------------------------------"
echo "Pushing image to Cloud Container Registry"
echo "------------------------------------------"
# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}


# Submit your training job
echo "----------------------------"
echo "Submitting the training job"
echo "----------------------------"


OUTPUT_LOCATION=gs://brain-ai-image-dev-artifacts/dali_usecase-image-similarity/training/jobs/$JOB_NAME

gcloud config set project ${PROJECT_ID}
gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --scale-tier CUSTOM \
    --master-machine-type=n1-standard-16 \
    --master-accelerator=count=1,type=NVIDIA_TESLA_P100 \
    --service-account=vertex-ai-training@brain-ai-image-dev.iam.gserviceaccount.com \
    -- \
    $JOB_NAME
 
    # these arguments will be passed to the train.sh
    #--master-accelerator=count=1,type=NVIDIA_TESLA_V100
    #--master-machine-type=n1-highmem-8 \

echo "------------------------------------------------------"
echo "Done submitting, check it on GCP AI Platform Dashboard"
echo "------------------------------------------------------"