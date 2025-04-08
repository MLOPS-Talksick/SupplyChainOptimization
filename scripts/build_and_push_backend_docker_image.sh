#!/usr/bin/env bash
set -e

# Configurable variables (fallbacks provided)
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-cloudrun_backend}"
IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
REPO_URI="${GCP_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "📦 Docker build triggered for backend (image missing or code changed)."
echo "📸 Building image: ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build from backend/ folder
docker build --no-cache -t "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" -f backend/Dockerfile backend

# Tag latest (optional but useful)
docker tag "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" "${REPO_URI}/${IMAGE_NAME}:latest"

# Push both tags
docker push "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
docker push "${REPO_URI}/${IMAGE_NAME}:latest"

echo "✅ Backend image pushed successfully:"
echo "   - ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"
echo "   - ${REPO_URI}/${IMAGE_NAME}:latest"

# Output for GitHub Actions
echo "backend_image_uri=${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" >> "$GITHUB_OUTPUT"