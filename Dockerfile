FROM nvcr.io/nvidia/pytorch:25.09-py3

# CUDA 환경 변수 설정
ENV CUDA_HOME=/usr/local/cuda-13.0
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include
ENV TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

# vLLM 관련 환경 변수
ENV VLLM_TARGET_DEVICE="cuda"
ENV MAX_JOBS=4

# 1. Install triton from source
RUN git clone https://github.com/triton-lang/triton.git && \
	cd triton && \
	git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
	pip install -r python/requirements.txt && \
	pip install . --no-build-isolation && \
	cd ..


# 2. Install xformers from source
# 문제 해결: PyTorch 2.9 호환성을 위해 v0.0.33 태그를 체크아웃
# --depth=1을 제거하고 전체를 clone한 뒤 특정 태그로 checkout
RUN git clone https://github.com/facebookresearch/xformers.git --recursive && \
	cd xformers && \
	git checkout v0.0.33 && \
	export TORCH_CUDA_ARCH_LIST="12.1" && \
	pip install . --no-deps --no-build-isolation && \
	cd ..

  
# 3. Install vLLM
RUN pip install "cmake>=3.21" ninja packaging setuptools-scm>=8 wheel jinja2

# vLLM 설치 (v0.12.0 - TRL 공식 지원 최신 버전)
# TRL은 vLLM v0.10.2, v0.11.x, v0.12.0만 공식 지원
# 참고: https://docs.vllm.ai/en/latest/training/trl/
# vLLM 패치 적용 (sed를 사용한 안정적인 아키텍처 제한)
# Blackwell(GB10) GPU에서의 특정 커널 컴파일 에러를 해결하기 위해 SM12.x 아키텍처를 제외
RUN git clone https://github.com/vllm-project/vllm.git && \
	cd vllm && \
	git checkout v0.13.0 && \
	sed -i 's/"10.0f;11.0f;12.0f"/"10.0f"/g' CMakeLists.txt && \
	sed -i 's/"10.0a;10.1a;12.0a;12.1a"/"10.0a;10.1a"/g' CMakeLists.txt && \
	sed -i 's/"10.0f;11.0f"/"10.0f"/g' CMakeLists.txt && \
	sed -i 's/"10.0a;10.1a;10.3a;12.0a;12.1a"/"10.0a;10.1a;10.3a"/g' CMakeLists.txt && \
	export TORCH_CUDA_ARCH_LIST="12.1" && \
    python use_existing_torch.py && \
    pip install -r requirements/build.txt && \
	pip install --no-build-isolation -e . && \
    cd ..


# 4. Install unsloth and other dependencies
RUN pip install --upgrade unsloth unsloth_zoo qwen-vl-utils wandb "transformers==4.57.6" && \
    pip install --no-deps "trl==0.24.0" "peft>=0.18.0" accelerate "bitsandbytes==0.45.5" && \
	pip install sentence-transformers

# 5. Download Tiktoken Encodings (For offline reliability)
ENV TIKTOKEN_ENCODINGS_BASE=/workspace/tiktoken_encodings
RUN mkdir -p ${TIKTOKEN_ENCODINGS_BASE} && \
    wget -O ${TIKTOKEN_ENCODINGS_BASE}/o200k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O ${TIKTOKEN_ENCODINGS_BASE}/cl100k_base.tiktoken \
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# 6. Performance & Network Environment Variables for Blackwell/DGX
ENV NCCL_IB_DISABLE=0
ENV NCCL_DEBUG=WARN
ENV NCCL_ASYNC_ERROR_HANDLING=1

# vLLM V1 Engine and Optimization Settings
ENV VLLM_USE_V1=1
ENV VLLM_ATTENTION_BACKEND=FLASHINFER
ENV VLLM_CUDA_GRAPH_MODE=full_and_piecewise
ENV VLLM_USE_FLASHINFER_MOE_FP4=1
ENV VLLM_FLASHINFER_MOE_BACKEND=latency

# Launch the shell
WORKDIR /workspace
CMD ["/bin/bash"]
