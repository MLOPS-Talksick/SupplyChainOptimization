# ML Models

## Overview

This directory contains machine learning models used for supply chain optimization. The models are designed to predict demand, optimize inventory levels, and improve overall supply chain efficiency.

## Files and Directories

- **scripts/**: Contains Python scripts for running and managing the machine learning models.
- **docker-compose.yml**: Defines and runs multi-container Docker applications for managing services required by the ML models.
- **requirements.txt**: Lists Python dependencies required to run the models, ensuring consistency across environments.
- **Dockerfile**: Used to create a Docker image for the application, enabling containerization and portability.
- **database.env**: Contains environment variables for database configuration, such as credentials and hostnames.
- **experiments/**: Contains experiments conducted to fine-tune the models, with documentation of configurations and results.

## Setup

1. **Environment Setup**

   - Ensure you have Python 3.8 or higher installed.
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
  python scripts/run_model.py --model demand_forecast
  ```

## Experiments

- Document experiments in the `experiments` directory.
- Include configuration and results for each experiment.

## Next Steps

1. Refine models based on new data.
2. Explore additional features to improve accuracy.
3. Document new experiments and outcomes.

## Appendix

- For troubleshooting, ensure all dependencies are correctly installed and Docker is running.
- Refer to the official documentation for further resources.
