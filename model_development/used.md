```gcloud pubsub topics create vertex-model-registry-updates```

### EventArc trigger chats
https://chatgpt.com/share/67eecf6e-ff98-8002-8ed9-1ba268b8a0b5



```bash
gcloud eventarc triggers create vertex-model-update-trigger --location=us-central1 --destination-run-service=model-serving --destination-run-region=us-central1 --destination-run-path="/reload-model" --event-filters="type=google.cloud.aiplatform.v1.ModelService.UpdateModel"
```