#!/usr/bin/env bash
set -e

# Configurable variables (fallbacks provided)
PROJECT_ID="${GCP_PROJECT_ID}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-cloudrun_model_health_check}"
IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
REPO_URI="${GCP_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "Docker build triggered for Model Health Check."
echo "Building image: ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build from ML_Models/scripts folder
docker build \
  --no-cache \
  -t "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" \
  -f ML_Models/scripts/Dockerfile \
  ML_Models/scripts

# Tag latest
docker tag \
  "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" \
  "${REPO_URI}/${IMAGE_NAME}:latest"

# Push both tags
docker push "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REPO_URI}/${IMAGE_NAME}:latest"

echo "Model Health Check image pushed successfully:"
echo "   - ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - ${REPO_URI}/${IMAGE_NAME}:latest"

# Export for GitHub Actions
echo "model_health_check_image_uri=${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" >> "$GITHUB_OUTPUT"