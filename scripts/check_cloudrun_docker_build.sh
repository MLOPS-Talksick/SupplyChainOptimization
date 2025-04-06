#!/usr/bin/env bash
set -e

# Use environment variables or fallback defaults.
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
# For model training, we assume a repository named "ml-models".
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-model_training}"
IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
LOCATION="${GCP_LOCATION:-us-central1}"

FULL_IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "🔍 Checking if '${IMAGE_NAME}:${IMAGE_TAG}' exists in Artifact Registry..."

# If the repository or image is not found, return an empty JSON array instead of failing.
IMAGES_JSON=$(gcloud artifacts docker images list "${FULL_IMAGE_PATH}" --include-tags --format="json" 2>/dev/null || echo "[]")

# Extract matching image package name.
MATCHING_IMAGE=$(echo "$IMAGES_JSON" | jq -r --arg IMAGE_NAME "$FULL_IMAGE_PATH" '.[] | select(.package==$IMAGE_NAME) | .package')

# Initialize flag for build requirement.
BUILD_REQUIRED=false

if [[ "$MATCHING_IMAGE" == "$FULL_IMAGE_PATH" ]]; then
  echo "✅ Exact match found: '${IMAGE_NAME}:${IMAGE_TAG}'"
else
  echo "⚠️ No exact match for '${IMAGE_NAME}:${IMAGE_TAG}'. A new build is required."
  BUILD_REQUIRED=true
fi

echo "🔍 Checking if files in 'model_development/model_training' have changed..."

# Check commit history; if not enough, assume changes.
if [ "$(git rev-list --count HEAD)" -lt 2 ]; then
  echo "⚠️ Not enough commit history. Assuming changes."
  BUILD_REQUIRED=true
else
  # Check for changes in the entire model_development/model_training folder.
  if ! git diff --quiet HEAD~1 HEAD -- model_development/model_training; then
    echo "⚠️ Changes detected in model_development/model_training. A new build is required."
    BUILD_REQUIRED=true
  fi
fi

# Write the build requirement status to GitHub Actions output.
if [[ "$BUILD_REQUIRED" == "true" ]]; then
  echo "Building Docker image from model_development/model_training/Dockerfile..."
  
  # Generate a new image tag based on current date and time.
  NEW_IMAGE_TAG=$(date +'%Y%m%d-%H%M%S')
  echo "Building image: ${FULL_IMAGE_PATH}:${NEW_IMAGE_TAG}"
  
  # Build the Docker image.
  docker build --no-cache -t "${FULL_IMAGE_PATH}:${NEW_IMAGE_TAG}" -f model_development/model_training/Dockerfile model_development/model_training
  
  # Also tag this build as 'latest' for convenience.
  docker tag "${FULL_IMAGE_PATH}:${NEW_IMAGE_TAG}" "${FULL_IMAGE_PATH}:latest"
  
  # Push both the new tag and the 'latest' tag.
  docker push "${FULL_IMAGE_PATH}:${NEW_IMAGE_TAG}"
  docker push "${FULL_IMAGE_PATH}:latest"
  
  echo "Image pushed successfully with tag: ${NEW_IMAGE_TAG}"
  echo "build_required=true" >> "$GITHUB_OUTPUT"
else
  echo "No build required."
  echo "build_required=false" >> "$GITHUB_OUTPUT"
fi
