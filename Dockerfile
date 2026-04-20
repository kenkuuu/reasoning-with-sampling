FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir \
    transformers==4.47.1 \
    accelerate==1.10.0 \
    datasets==3.2.0 \
    pandas==2.2.3 \
    tqdm==4.67.1 \
    sentencepiece==0.2.0 \
    tiktoken==0.7.0 \
    safetensors==0.4.5

WORKDIR /workspace
COPY llm_experiments/ llm_experiments/
