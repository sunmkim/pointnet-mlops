FROM pytorch/pytorch:2.14.1-cuda13.2-cudnn9-runtime

# Prevent interactive prompts and Python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN ["uv", "run"]