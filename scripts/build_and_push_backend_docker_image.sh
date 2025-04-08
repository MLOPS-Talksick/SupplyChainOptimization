# scripts/build_and_push_backend_docker_image.sh
#!/usr/bin/env bash
set -e

PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="airflow-docker-image"
IMAGE_NAME="cloudrun_backend"
IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
GCP_LOCATION="${GCP_LOCATION:-us-central1}"
REPO_URI="${GCP_LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}"

echo "📦 Building backend image: ${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

docker build --no-cache -t "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .
docker push "${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "backend_image_uri=${REPO_URI}/${IMAGE_NAME}:${IMAGE_TAG}" >> "$GITHUB_OUTPUT"
