<pre> ```mermaid flowchart TD A[Data Generation] --> B[Data Pipeline] B --> C[ML Models] C --> D[Deployment] D --> E[Monitoring] click A href "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Generation" "Data Generation" click B href "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Pipeline" "Data Pipeline" click C href "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/ML_Models" "ML Models" click D href "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Deployment" "Deployment" click E href "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Monitoring" "Monitoring" ``` </pre>



flowchart TD
    %% Data Generation
    subgraph A[Data Generation]
        A1[dataGenerator.py]
        A2[transactionGenerator.py]
    end

    %% Data Pipeline
    subgraph B[Data Pipeline]
        B1[Preprocessing]
        B2[Feature Engineering]
        B3[Data Validation - Great Expectations]
        B4[DVC for Data Versioning]
    end

    %% Orchestration
    subgraph C[Workflow Orchestration]
        C1[Apache Airflow DAGs]
    end

    %% Machine Learning
    subgraph D[ML Models]
        D1[Training Scripts]
        D2[Evaluation Scripts]
        D3[MLflow for Tracking]
    end

    %% Deployment
    subgraph E[Deployment]
        E1[Docker & Compose]
        E2[Terraform Infra on GCP]
        E3[Backend APIs]
        E4[Frontend App]
    end

    %% Monitoring & Feedback
    subgraph F[Monitoring & Feedback]
        F1[Performance Logs]
        F2[Drift Detection]
        F3[Auto Retraining]
    end

    %% Flow connections
    A --> B
    B --> B4 --> C --> D
    D --> E
    E --> F
    F3 --> B
    F1 --> D3

    %% Clickable URLs
    click A1 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/Data_Generation/dataGenerator.py" "View dataGenerator.py"
    click B4 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/Data_Pipeline" "Go to Data Pipeline"
    click C1 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/dags" "View Airflow DAGs"
    click D3 "https://github.com/MLOPS-Talksick/SupplyChainOptimization" "MLflow Tracking"
    click E1 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/blob/main/docker-compose.yaml" "Docker Setup"
    click E3 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/backend" "Backend Code"
    click E4 "https://github.com/MLOPS-Talksick/SupplyChainOptimization/tree/main/frontend" "Frontend Code"
