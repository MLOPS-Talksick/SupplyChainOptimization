-> Build the docker file ```docker build -f model_development\model_training_cloud_run_trigger\Dockerfile -t model_training_trigger .```
-> List docker images ```docker images```
-> Tag the docker file ```docker tag model_training_trigger us-central1-docker.pkg.dev/primordial-veld-450618-n4/ml-models/model_training_trigger:latest```
-> Push the docker file ```docker push us-central1-docker.pkg.dev/primordial-veld-450618-n4/ml-models/model_training_trigger```