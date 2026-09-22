"""Deploy the OmniVoice TTS SageMaker endpoint (omnivoice-tts).

Run: uv run python sagemaker/deploy_tts.py

Requires:
- AWS credentials configured
- SageMaker execution role with S3 access
"""

import contextlib
import os
import tarfile
import tempfile
import time

import boto3

# Configuration
ROLE = os.environ.get("SAGEMAKER_ROLE")
INSTANCE_TYPE = "ml.g5.2xlarge"
ENDPOINT_NAME = "omnivoice-tts"
REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
BUCKET = os.environ.get("SAGEMAKER_BUCKET", None)

# HuggingFace DLC image for PyTorch inference
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md
# Generic PyTorch DLC (no HuggingFace toolkit — avoids transformers version conflicts)
HF_IMAGE = (
    f"763104351884.dkr.ecr.{REGION}.amazonaws.com"
    f"/pytorch-inference:2.6.0-gpu-py312-cu124-ubuntu22.04-sagemaker-v1.81"
)


def create_model_tar():
    """Create model.tar.gz with inference code and reference audio."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tar_path = f.name

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add("sagemaker/tts_handler.py", arcname="code/tts_handler.py")
        tar.add("sagemaker/requirements.txt", arcname="code/requirements.txt")

        if os.path.exists("assets/voice_ref_th.wav"):
            tar.add("assets/voice_ref_th.wav", arcname="voice_ref_th.wav")

    print(f"  Created {tar_path}")
    return tar_path


def deploy():
    if not ROLE:
        raise SystemExit(
            "SAGEMAKER_ROLE is not set. Export the ARN of a SageMaker execution "
            "role, e.g.\n  export SAGEMAKER_ROLE=arn:aws:iam::<account>:role/<role>"
        )

    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")
    s3 = session.client("s3")

    bucket = BUCKET
    if not bucket:
        sts = session.client("sts")
        account = sts.get_caller_identity()["Account"]
        bucket = f"sagemaker-{REGION}-{account}"
        # Create bucket if it doesn't exist
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:
            print(f"  Creating bucket {bucket}...")
            if REGION == "us-east-1":
                s3.create_bucket(Bucket=bucket)
            else:
                s3.create_bucket(
                    Bucket=bucket,
                    CreateBucketConfiguration={"LocationConstraint": REGION},
                )

    # 1. Create and upload model archive
    print("[1/4] Creating model archive...")
    tar_path = create_model_tar()

    s3_key = "omnivoice-tts/model.tar.gz"
    print(f"[2/4] Uploading to s3://{bucket}/{s3_key}...")
    s3.upload_file(tar_path, bucket, s3_key)
    os.unlink(tar_path)
    model_data_url = f"s3://{bucket}/{s3_key}"

    # 2. Create model
    model_name = f"{ENDPOINT_NAME}-model"
    print(f"[3/4] Creating model '{model_name}'...")

    # Delete existing model if any
    with contextlib.suppress(Exception):
        sm.delete_model(ModelName=model_name)

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": HF_IMAGE,
            "ModelDataUrl": model_data_url,
            "Environment": {
                "SAGEMAKER_PROGRAM": "tts_handler.py",
                "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code",
                "SAGEMAKER_MODEL_SERVER_TIMEOUT": "600",
            },
        },
        ExecutionRoleArn=ROLE,
    )

    # 3. Create endpoint config
    config_name = f"{ENDPOINT_NAME}-config"
    with contextlib.suppress(Exception):
        sm.delete_endpoint_config(EndpointConfigName=config_name)

    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "primary",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "ContainerStartupHealthCheckTimeoutInSeconds": 600,
                "ModelDataDownloadTimeoutInSeconds": 600,
            }
        ],
    )

    # 4. Create or update endpoint
    print(f"[4/4] Deploying endpoint '{ENDPOINT_NAME}' on {INSTANCE_TYPE}...")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        # Endpoint exists, update it
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        print("  Updating existing endpoint...")
    except sm.exceptions.ClientError:
        # Create new endpoint
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=config_name,
        )
        print("  Creating new endpoint...")

    # Wait for endpoint
    print("  Waiting for endpoint to be InService (this takes 5-10 minutes)...")
    while True:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        if status == "InService":
            break
        elif status == "Failed":
            print(f"  FAILED: {resp.get('FailureReason', 'unknown')}")
            return
        print(f"  Status: {status}...")
        time.sleep(30)

    print(f"\nEndpoint '{ENDPOINT_NAME}' is InService!")
    print("\nTest with: uv run python sagemaker/test_endpoints.py")


if __name__ == "__main__":
    deploy()
