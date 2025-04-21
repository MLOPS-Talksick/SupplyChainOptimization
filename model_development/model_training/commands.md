-> Run this from the root: ```docker build -f model_development\model_training\Dockerfile -t model_training .```

-> ```docker tag model_training us-central1-docker.pkg.dev/primordial-veld-450618-n4/ml-models/model_training:latest```
-> Push the docker file ```docker push us-central1-docker.pkg.dev/primordial-veld-450618-n4/ml-models/model_training```


files copied:
├── Data_Pipeline/
│   └── scripts
│   │       ├── logger.py
│   │        └── utils.py
│   └── database.env
│
├── ML_Models/  
│    └── scripts
│           ├── utils.py
│           └── model_xgboost.py
│
├── model_development/ 
│    └── model_training
│           ├── Dockerfile
│           └── requirements.txt