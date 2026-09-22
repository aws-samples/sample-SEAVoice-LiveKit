"""Deploy Qwen3-ASR to a SageMaker endpoint using the AWS vLLM DLC.

    export SAGEMAKER_ROLE=arn:aws:iam::<account>:role/<your-sagemaker-role>
    uv run python sagemaker/deploy_stt.py

Serves Qwen3-ASR on vLLM via AWS's SageMaker vLLM Deep Learning Container. The image
is pulled from ECR Public through an ECR pull-through cache (created here if missing),
so no Docker or image build is needed. Cloud TTS is served by OmniVoice — see deploy_tts.py.

Prerequisites (one-time, see sagemaker/README.md):
  - AWS credentials + a SageMaker execution role (SAGEMAKER_ROLE).
  - The execution role needs ECR pull-through permissions on the cached repos:
      ecr:BatchImportUpstreamImage, ecr:CreateRepository, ecr:BatchGetImage,
      ecr:GetDownloadUrlForLayer  (resource: .../repository/ecr-public/*)
  - GPU quota for ml.g5.2xlarge endpoint usage.
"""

import contextlib
import os
import time

import boto3

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
ENDPOINT_NAME = "qwen3-asr"
MODEL_ID = "Qwen/Qwen3-ASR-0.6B"
INSTANCE_TYPE = "ml.g5.2xlarge"
# CUDA-13 images need this inference AMI; the default host AMI's driver is too old.
INFERENCE_AMI_VERSION = "al2-ami-sagemaker-inference-gpu-3-1"
PULL_THROUGH_PREFIX = "ecr-public"


def _resolve_role() -> str:
    role = os.environ.get("SAGEMAKER_ROLE")
    if not role:
        raise SystemExit(
            "SAGEMAKER_ROLE is not set. Export the ARN of a SageMaker execution role."
        )
    return role


def _ensure_pull_through_cache(session) -> str:
    """Ensure an ECR pull-through cache rule for ECR Public exists; return the account."""
    account = session.client("sts").get_caller_identity()["Account"]
    ecr = session.client("ecr")
    rules = ecr.describe_pull_through_cache_rules()["pullThroughCacheRules"]
    if not any(r["ecrRepositoryPrefix"] == PULL_THROUGH_PREFIX for r in rules):
        print(f"  Creating ECR pull-through cache rule '{PULL_THROUGH_PREFIX}'...")
        ecr.create_pull_through_cache_rule(
            ecrRepositoryPrefix=PULL_THROUGH_PREFIX,
            upstreamRegistryUrl="public.ecr.aws",
        )
    return account


def deploy() -> None:
    role = _resolve_role()
    session = boto3.Session(region_name=REGION)
    sm = session.client("sagemaker")

    account = _ensure_pull_through_cache(session)
    image = (
        f"{account}.dkr.ecr.{REGION}.amazonaws.com/"
        f"{PULL_THROUGH_PREFIX}/deep-learning-containers/vllm:server-sagemaker-cuda"
    )

    model_name = f"{ENDPOINT_NAME}-model"
    config_name = f"{ENDPOINT_NAME}-config"
    with contextlib.suppress(Exception):
        sm.delete_model(ModelName=model_name)
    with contextlib.suppress(Exception):
        sm.delete_endpoint_config(EndpointConfigName=config_name)

    print(f"[1/3] Creating model '{model_name}' (vLLM DLC, {MODEL_ID})...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image,
            "Environment": {
                "SM_VLLM_MODEL": MODEL_ID,
                "SM_VLLM_GPU_MEMORY_UTILIZATION": "0.7",
            },
        },
        ExecutionRoleArn=role,
    )

    print("[2/3] Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[
            {
                "VariantName": "default",
                "ModelName": model_name,
                "InstanceType": INSTANCE_TYPE,
                "InitialInstanceCount": 1,
                "InferenceAmiVersion": INFERENCE_AMI_VERSION,
                "ContainerStartupHealthCheckTimeoutInSeconds": 900,
            }
        ],
    )

    print(f"[3/3] Deploying endpoint '{ENDPOINT_NAME}' on {INSTANCE_TYPE}...")
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        print("  Updating existing endpoint...")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
        print("  Creating new endpoint...")

    print("  Waiting for InService (first pull-through of the image is slow)...")
    while True:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp["EndpointStatus"]
        if status == "InService":
            break
        if status == "Failed":
            print(f"  FAILED: {resp.get('FailureReason', 'unknown')}")
            return
        print(f"  Status: {status}...")
        time.sleep(30)

    print(f"\nEndpoint '{ENDPOINT_NAME}' is InService!")


if __name__ == "__main__":
    deploy()
