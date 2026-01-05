# Docker Image: gogamza/unsloth-vllm-gb10:latest

This Docker image is purpose-built for high-performance training and inference using **Unsloth** and **vLLM** on **NVIDIA Blackwell (SM 10.0/GB10)** GPU architectures. Based on NVIDIA's latest PyTorch container, it includes custom-built binaries and patches to ensure seamless compatibility and maximum throughput for Large Language Models (LLM) and Vision-Language Models (VLM).

### 🚀 Key Features

*   **Blackwell (SM 10.0) Optimization**: Specifically patched for Blackwell GPU architectures to resolve kernel compilation issues. Built with `TORCH_CUDA_ARCH_LIST="12.1"` compatibility.
*   **vLLM V1 Engine**: Enables high-throughput inference using the latest `VLLM_USE_V1=1` engine with **FlashInfer** as the primary attention backend.
*   **FP4 Precision Support**: Ready for high-efficiency MoE (Mixture of Experts) models with `VLLM_USE_FLASHINFER_MOE_FP4` support.
*   **Unsloth Full Stack**: Comes pre-installed with `unsloth`, `unsloth_zoo`, and `qwen-vl-utils`, making it ready for fine-tuning models like Qwen3-VL out of the box.
*   **Offline Reliability**: Pre-cached `tiktoken` encoding files (`o200k`, `cl100k`) to ensure stability in air-gapped or restricted network environments.

### 🛠 Tech Stack

*   **Base Image**: `nvcr.io/nvidia/pytorch:25.09-py3`
*   **CUDA**: 13.0
*   **vLLM**: Built from source with Blackwell-specific patches.
*   **xformers**: v0.0.33 (Built from source)
*   **Triton**: Custom commit (c5d671f) for optimized kernel execution.
*   **Tools**: Pre-configured FlashInfer and advanced environment optimizations.

### 💻 Quick Start

You can launch the container on a Blackwell-ready environment as follows:

```bash
docker run --gpus all -it --rm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  gogamza/unsloth-vllm-gb10:latest
```

### ⚙️ Pre-configured Environment Variables

The following optimizations are enabled by default:
- `VLLM_USE_V1=1`
- `VLLM_ATTENTION_BACKEND=FLASHINFER`
- `VLLM_CUDA_GRAPH_MODE=full_and_piecewise`
- `NCCL_IB_DISABLE=0` (Optimized for DGX/Blackwell inter-GPU communication)
