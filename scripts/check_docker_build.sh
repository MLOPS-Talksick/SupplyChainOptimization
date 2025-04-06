#!/usr/bin/env bash
set -e

# Use environment variables (set in GitHub Actions or your local environment) or fallback defaults.
PROJECT_ID="${GCP_PROJECT_ID:-primordial-veld-450618-n4}"
REPO_NAME="${ARTIFACT_REGISTRY_NAME:-airflow-docker-image}"
IMAGE_NAME="${DOCKER_IMAGE_NAME:-data-pipeline}"
IMAGE_TAG="${DOCKER_IMAGE_TAG:-latest}"
LOCATION="${GCP_LOCATION:-us-central1}"

FULL_IMAGE_PATH="${LOCATION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}"

echo "🔍 Checking if '${IMAGE_NAME}:${IMAGE_TAG}' exists in Artifact Registry..."

# If the repository or image is not found, return an empty JSON array instead of failing.
IMAGES_JSON=$(gcloud artifacts docker images list "${FULL_IMAGE_PATH}" --include-tags --format="json" 2>/dev/null || echo "[]")

# Initialize flag for build requirement.
BUILD_REQUIRED=false

# Check for image and exact tag match
MATCH_FOUND=$(echo "$IMAGES_JSON" | jq -r --arg IMAGE "$FULL_IMAGE_PATH" --arg TAG "$IMAGE_TAG" '
  .[] | select(.package == $IMAGE and (.tags[]? == $TAG)) | .package')

if [[ "$MATCH_FOUND" == "$FULL_IMAGE_PATH" ]]; then
  echo "✅ Exact image and tag match found: '${IMAGE_NAME}:${IMAGE_TAG}'"
else
  echo "⚠️ Image or tag not found. A new build is required."
  BUILD_REQUIRED=true
fi

echo "🔍 Checking if 'Dockerfile' or 'requirements.txt' has changed..."

# Only run git diff if there’s a previous commit to compare to
if git rev-parse HEAD~1 >/dev/null 2>&1; then
  if git diff --quiet HEAD~1 HEAD -- Data_Pipeline/Dockerfile Data_Pipeline/requirements.txt; then
    echo "✅ No changes detected in 'Dockerfile' or 'requirements.txt'."
  else
    echo "⚠️ Changes detected in 'Dockerfile' or 'requirements.txt'. A new build is required."
    BUILD_REQUIRED=true
  fi
else
  echo "ℹ️ Only one commit found. Skipping change detection."
fi

# Write the build requirement status to GitHub Actions output.
if [[ "$BUILD_REQUIRED" == "true" ]]; then
  echo "build_required=true" >> "$GITHUB_OUTPUT"
else
  echo "build_required=false" >> "$GITHUB_OUTPUT"
fi