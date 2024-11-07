## Video Frame Extraction Pipeline Setup Guide
This guide will walk you through the one-time setup required to deploy and run a Vertex AI pipeline that extracts frames from videos using a custom Docker image.

### Project Information
- Project Name: eh-ml-mentorship
- Project ID: racoon-detection-427421
- Project Number: 461769263335
- Region: us-central1

### Table of Contents
Prerequisites
Step 1: Enable Required APIs
Step 2: Install and Configure Google Cloud SDK
Step 3: Set Up Python Environment
Step 4: Configure Artifact Registry Access
Step 5: Define the Custom Component
Step 6: Write the Pipeline Code
Step 7: Compile the Pipeline
Step 8: Set Up IAM Permissions
Step 9: Submit the Pipeline to Vertex AI
Step 10: Run the Pipeline
Additional Notes
Conclusion

### Prerequisites
Google Cloud Account: Access to the eh-ml-mentorship project.
Local Machine Setup: Ability to run commands in a terminal and install software.

```sh
gcloud config set project eh-ml-mentorship
gcloud services enable aiplatform.googleapis.com
gcloud services enable artifactregistry.googleapis.com
gcloud services enable storage.googleapis.com
gcloud projects add-iam-policy-binding eh-ml-mentorship \
  --member=serviceAccount:461769263335-compute@developer.gserviceaccount.com \
  --role=roles/artifactregistry.reader

gcloud artifacts repositories add-iam-policy-binding video-frame-extractor-repo \
--location=us-central1 \
--member=297238540166-compute@developer.gserviceaccount.com\
--role=roles/artifactregistry.reader \
--condition=None

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install kfp google-cloud-pipeline-components google-cloud-aiplatform
python pipeline.py
python submit_pipeline.py

```