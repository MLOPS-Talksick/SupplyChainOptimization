#!/usr/bin/env bash
set -e

# Use environment variables or fallback defaults.
PROJECT_ID="${GCP_PROJECT_ID}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-model_training}"
IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
REPO_URI="${GCP_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "📦 Docker build triggered (image missing or code changed)."
echo "📸 Building image: ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

# ✅ Build Docker image from project root so all folders are accessible
docker build --no-cache -t "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" -f model_development/model_training/Dockerfile .

# ✅ Tag the build as 'latest'
docker tag "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" "${REPO_URI}/${IMAGE_NAME}:latest"

# ✅ Push both timestamped and latest tag
docker push "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REPO_URI}/${IMAGE_NAME}:latest"

echo "✅ Image pushed successfully:"
echo "   - ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - ${REPO_URI}/${IMAGE_NAME}:latest"
echo "model_training_image_uri=${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" >> "$GITHUB_OUTPUT"