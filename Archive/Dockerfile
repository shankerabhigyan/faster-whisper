FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV PYTHONUNBUFFERED=1 
ENV DEBIAN_FRONTEND noninteractive

# Prerequisites
RUN apt-get update -y -q \
    && apt-get install -y -q --no-install-recommends \
        software-properties-common \
        build-essential apt-utils \
        wget curl vim git ca-certificates kmod gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Python 3.9
RUN add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update --yes --quiet
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes --quiet --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 999 \
    && update-alternatives --config python3 && ln -s /usr/bin/python3 /usr/bin/python \
    && pip install --upgrade pip

COPY app app
COPY requirements.txt requirements.txt

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt \
    && python3 -m pip install --no-cache-dir torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117

WORKDIR app

ENTRYPOINT ["python3", "server.py"]