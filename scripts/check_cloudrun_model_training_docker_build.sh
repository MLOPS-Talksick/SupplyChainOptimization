#!/usr/bin/env bash
set -euo pipefail

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-model_training}"
IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
LOCATION="${GCP_LOCATION:-us-central1}"

FULL_IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "🔍 Checking if '${IMAGE_NAME}:${IMAGE_TAG}' exists in Artifact Registry..."
IMAGES_JSON=$(gcloud artifacts docker images list "${FULL_IMAGE_PATH}" \
  --include-tags --format="json" 2>/dev/null || echo "[]")

BUILD_REQUIRED=false

MATCH_FOUND=$(echo "$IMAGES_JSON" | jq -r --arg IMAGE "$FULL_IMAGE_PATH" --arg TAG "$IMAGE_TAG" \
  '.[] | select(.package == $IMAGE and (.tags[]? == $TAG)) | .package')

if [[ "$MATCH_FOUND" == "$FULL_IMAGE_PATH" ]]; then
  echo "✅ Exact image and tag match found: '${IMAGE_NAME}:${IMAGE_TAG}'"
else
  echo "⚠️ Image or tag not found. A new build is required."
  BUILD_REQUIRED=true
fi

# Watch all of your training code directories, not just model_development/model_training
CODE_DIRS=(
  "model_development/model_training"
  "ML_Models/scripts"
  "Data_Pipeline/scripts"
)

echo "🔍 Checking if files in training dirs have changed..."
if git rev-parse HEAD~1 >/dev/null 2>&1; then
  for dir in "${CODE_DIRS[@]}"; do
    if ! git diff --quiet HEAD~1 HEAD -- "$dir"; then
      echo "⚠️ Changes detected in last commit under '$dir'. A new build is required."
      BUILD_REQUIRED=true
      break
    fi
  done
else
  echo "ℹ️ Only one commit found. Checking working directory changes instead..."
  for dir in "${CODE_DIRS[@]}"; do
    if ! git diff --quiet -- "$dir"; then
      echo "⚠️ Uncommitted or staged changes found under '$dir'. A new build is required."
      BUILD_REQUIRED=true
      break
    fi
  done
fi

echo "build_required=${BUILD_REQUIRED}" >> "$GITHUB_OUTPUT"