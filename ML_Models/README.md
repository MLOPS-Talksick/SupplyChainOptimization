# ML Models

## Overview

This directory contains machine learning models used for supply chain optimization. The models are designed to predict demand, optimize inventory levels, and improve overall supply chain efficiency.

## Files and Directories

- **scripts/**: Contains Python scripts for running and managing the machine learning models.
- **docker-compose.yml**: Defines and runs multi-container Docker applications for managing services required by the ML models.
- **requirements.txt**: Lists Python dependencies required to run the models, ensuring consistency across environments.
- **Dockerfile**: Used to create a Docker image for the application, enabling containerization and portability.
- **database.env**: Contains environment variables for database configuration, such as credentials and hostnames. Obtain from original contributers.
- **experiments/**: Contains experiments conducted to fine-tune the models, with documentation of configurations and results.

## Setup

1. **Environment Setup**

   - Ensure you have Python 3.10 installed.
   - Use a virtual environment to manage dependencies.
   - Install the required packages:
     ```bash
     pip install -r requirements.txt
     ```

2. **Docker Setup**
   - Build the Docker image:
     ```bash
     docker build -t supplychain-ml-models .
     ```
   - Use `docker-compose` to manage applications:
     ```bash
     docker-compose up
     ```

## Usage

- Run models using scripts in the `scripts` directory.
- Example command:
  ```bash
  python scripts/model_xgboost.py
  ```

## Experiments

- Document experiments in the `experiments` directory.
- Include configuration and results for each experiment.


## Appendix

- For troubleshooting, ensure all dependencies are correctly installed and Docker is running.
- Refer to the official documentation for further resources.

## Meeting the assignment requirements:
1. Docker or RAG Format: The entire pipeline has been implemented in Docker format. This ensures reproducibility and
portability of your ML models.
2. Code for Loading Data from Data Pipeline: We have written code that loads
the data output from the data pipeline which in our case is an SQL table, ensuring all transformations and
versioning are applied as needed.
3. Code for Training Model and Selecting Best Model: We have included code that trains the model and selects the best model based on validation metrics. This is done using a popular optimization and hyperparameter tuning tool called optuna. Our experiments have proven that XGBoost performs significantly better in our use case as compared to other techniques such as LSTMs, DeepAR and SARIMA. Due to this we are using only XGBoost.
4. Code for Model Validation: We have written code that validates the model on a
separate validation dataset and computes metrics relevant to your task such as RMSE. This code is part of the train_xgboost.py file.
5. Code for Bias Checking: We have developed code to check for bias using data slicing techniques by specifically getting RMSE values on specific products. We also create SHAP plots while our feature space includes a one hot encoded vector of the products showing the effect a given product has on the output giving a certain degree of bias vizualisation. On top of this, if our model performs worse than 2 standard deveation of the mean, we create a hybrid model by training an XGBoost model on the bias product and combining it with the old model (meta-learner). This ensures that any bas is mitigated (technique confirmed with TA).
6. Code for Model Selection after Bias Checking: Point discussed above
7. Code to Push Model to Artifact Registry/Model Registry: This has been included. We are pushing code to GCS as it fits better with our vision of the next assignment.
8. Rollback mechanism: Code ensures that newer model performs better than current model. As model retrianing will only be orchestrated when rmse is worse than 20 or if drift is detected.