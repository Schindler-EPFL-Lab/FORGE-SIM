FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
# Set environment variables.
## Set non-interactive to prevent asking for user inputs blocking image creation.
ENV DEBIAN_FRONTEND=noninteractive
## Set timezone as it is required by some packages.
ENV TZ=Europe/Berlin
## CUDA Home, required to find CUDA in some packages.
ENV CUDA_HOME="/usr/local/cuda"
## Graphics driver capability (graphics required for OptiX)
ENV NVIDIA_DRIVER_CAPABILITIES="compute,utility,graphics"

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    curl \
    ffmpeg \
    git \
    llvm \
    llvm-dev \
    clang \
    clang-tools \
    libclang-dev \
    lldb \
    lld \
    nano \
    protobuf-compiler \
    qtbase5-dev \
    sudo \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*


RUN apt update -y
RUN apt install -y --no-install-recommends curl ca-certificates
RUN apt install -y python3-pip
RUN pip install uv --break-system-packages
RUN uv --version 

# Ensure sudo group users are not asked for a password when using sudo command by ammending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# Switch to new user and workdir.
USER 1000
WORKDIR /home/ubuntu
# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/ubuntu/.local/bin"
SHELL ["/bin/bash", "-c"]

# Created with pip freeze -l > requirements.txt
COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY .python-version .python-version
COPY README.md README.md
RUN ls

# Install dependencies
RUN uv sync --frozen --extra azure --extra depth-estimation

RUN uv pip freeze
# Put the python binary at the front of the path to use directly in future commands.
ENV PATH="/home/ubuntu/.venv/bin:$PATH"
# Change working directory
WORKDIR /workspace
 