#!/bin/bash
set -e

echo "=========================================================="
echo "  FIDELITY.AI — AMD INSTINCT MI300X vLLM DEPLOYMENT"
echo "=========================================================="

# 1. Open firewall port 8000
ufw allow 8000 || true

# 2. Stop old vllm container if running
docker rm -f vllm 2>/dev/null || true

# 3. Pull official AMD ROCm vLLM Docker container
echo "[1/3] Pulling optimized AMD ROCm vLLM image..."
docker pull rocm/vllm:latest

# 4. Launch vLLM server with Qwen 2.5-VL on MI300X
echo "[2/3] Launching Qwen 2.5-VL on AMD Instinct MI300X (192GB VRAM)..."
docker run -d \
  --name vllm \
  --network=host \
  --device=/dev/kfd \
  --device=/dev/dri \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --restart unless-stopped \
  rocm/vllm:latest \
  vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90

echo "[3/3] vLLM container launched successfully on port 8000!"
echo "Displaying real-time startup logs (press Ctrl+C anytime to detach):"
sleep 2
docker logs -f vllm
