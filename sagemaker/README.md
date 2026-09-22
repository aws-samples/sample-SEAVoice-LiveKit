# SageMaker Voice-interface Endpoints

The cloud speech path runs on two SageMaker endpoints that serve any language. Thai also ships local models (the default `telco_th`/`chat_th`), so it's the one language with a no-endpoint option, although you can use the `*_th_sagemaker` configs to run Thai here too.

## Endpoints

| Endpoint | Model / image | Used for | Deploy |
|----------|---------------|----------|--------|
| `qwen3-asr` | Qwen3-ASR on the AWS vLLM DLC | STT | `deploy_stt.py` |
| `omnivoice-tts` | k2-fsa/OmniVoice (PyTorch DLC) | TTS (646 languages) | `deploy_tts.py` |

STT uses AWS's SageMaker vLLM Deep Learning Container (`vllm:server-sagemaker-cuda`), which serves Qwen3-ASR fast (~0.3-0.6s/call vs ~1.5s for plain transformers) and needs no custom code. The image is pulled from ECR Public via an ECR pull-through cache (created automatically), so no Docker or image build is needed. Qwen3-ASR covers all six languages with auto-detect as a fallback.

## Prerequisites

1. **A SageMaker execution role**, exported as `SAGEMAKER_ROLE`:

   ```bash
   export SAGEMAKER_ROLE=arn:aws:iam::<account>:role/<your-sagemaker-role>
   export AWS_DEFAULT_REGION=us-west-2
   ```

2. **ECR pull-through permissions on the execution role** (so SageMaker can pull the
   vLLM DLC through the cache on first use). Attach a policy allowing, on resource
   `arn:aws:ecr:<region>:<account>:repository/ecr-public/*`:
   `ecr:BatchImportUpstreamImage`, `ecr:CreateRepository`, `ecr:BatchGetImage`,
   `ecr:GetDownloadUrlForLayer` (plus `ecr:GetAuthorizationToken` on `*`).

3. **GPU quota.** Each endpoint uses one `ml.g5.2xlarge`. Check *"ml.g5.2xlarge for
   endpoint usage"* in Service Quotas and request an increase if needed (both = 2).

## Deploy

```bash
uv run python sagemaker/deploy_stt.py    # qwen3-asr  (vLLM DLC via pull-through cache)
uv run python sagemaker/deploy_tts.py         # omnivoice-tts
```

The Qwen3-ASR endpoint's first deploy is slow (~15 min) because the vLLM DLC is pulled through the cache for the first time; later deploys are faster.

## Test

```bash
uv run python sagemaker/test_endpoints.py   # smoke-tests both (STT + TTS)
```

## Cost and teardown

`ml.g5.2xlarge` endpoints bill continuously (~$1.5/hr each) until deleted. Delete them when you're done:

```bash
for ep in qwen3-asr omnivoice-tts; do
  aws sagemaker delete-endpoint --endpoint-name "$ep" --region us-west-2
done
```

## Files

- `deploy_stt.py` — deploy `qwen3-asr` (AWS vLLM DLC; no custom code — just an env var).
- `deploy_tts.py` + `tts_handler.py` — deploy/serve `omnivoice-tts` (PyTorch DLC + handler).
- `test_endpoints.py` — smoke-test both endpoints (STT + TTS).

OmniVoice voice-clones from a bundled Thai reference clip (`assets/voice_ref_th.wav`), so the output timbre is that speaker regardless of language. Swap the reference clip for a different voice.
