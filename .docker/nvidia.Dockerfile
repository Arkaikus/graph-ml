# Base Image
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install dependencies
RUN apt update && apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev wget curl llvm libncurses5-dev xz-utils libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev git

RUN curl https://pyenv.run | sh
ENV HOME  /root/
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.11 --verbose
RUN pyenv global 3.11

# Install PyTorch
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Entrypoint
ENTRYPOINT ["python", "-c", "import torch; print(torch.cuda.is_available())"]

