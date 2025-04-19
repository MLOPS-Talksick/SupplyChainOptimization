from google.cloud import aiplatform
from flask import Flask, request, jsonify
# from dotenv import load_dotenv
import os
from flask_cors import CORS


app = Flask(__name__)

CORS(app)

project_id=os.environ.get('PROJECT_ID')

@app.route("/trigger-training", methods=["POST"])
def trigger_training():

    data = request.get_json()
    project_id = data.get('PROJECT_ID')
    region = data.get('REGION')
    staging_bucket_uri = data.get('BUCKET_URI')
    image_uri = data.get('IMAGE_URI')
    sa_email  = os.environ["TRAINING_SERVICE_ACCOUNT"]

    aiplatform.init(project=project_id, location=region, staging_bucket=staging_bucket_uri)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="custom-lstm-model-training",
        container_uri=image_uri,
    )


    model = job.run(
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="ACCELERATOR_TYPE_UNSPECIFIED",
        accelerator_count=0,
        sync=False,
        service_account=sa_email,
        env={
          "MYSQL_HOST":     os.environ["MYSQL_HOST"],
          "MYSQL_USER":     os.environ["MYSQL_USER"],
          "MYSQL_PASSWORD": os.environ["MYSQL_PASSWORD"],
          "MYSQL_DATABASE": os.environ["MYSQL_DATABASE"],
          "INSTANCE_CONN_NAME":   os.environ["INSTANCE_CONN_NAME"],
        },
    )
    return jsonify({"message": "Training started"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
