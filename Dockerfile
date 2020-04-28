FROM nvidia/opengl:1.0-glvnd-runtime-ubuntu18.04

################################## JUPYTERLAB ##################################

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get -o Acquire::ForceIPv4=true update && apt-get -yq dist-upgrade \
 && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
	locales cmake git build-essential software-properties-common python3.7 \
        python-pip \
	python3-pip python3-setuptools \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Add python 3.6 & python 3.7 to update-alternatives
RUN rm /usr/bin/python3 && ln -s python3.7 /usr/bin/python3

RUN pip3 install --upgrade pip setuptools \
 && python3 -m pip install jupyterlab==0.35.4 bash_kernel==0.7.1 tornado==5.1.1 \
 && python3 -m bash_kernel.install

ENV SHELL=/bin/bash \
	NB_USER=jovyan \
	NB_UID=1000 \
	LANG=en_US.UTF-8 \
	LANGUAGE=en_US.UTF-8

ENV HOME=/home/${NB_USER}

RUN adduser --disabled-password \
	--gecos "Default user" \
	--uid ${NB_UID} \
	${NB_USER}

EXPOSE 8888

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''"]

###################################### CUDA ####################################

RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDA_VERSION 10.0.130

ENV CUDA_PKG_VERSION 10-0=$CUDA_VERSION-1
# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-0=410.48-1 && \
    ln -s cuda-10.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/cuda/bin:${PATH}

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.0 brand=tesla,driver>=384,driver<385"

ENV NCCL_VERSION 2.4.2

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-nvtx-$CUDA_PKG_VERSION \
        libnccl2=$NCCL_VERSION-1+cuda10.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

##################################### APT ######################################

RUN apt-get -o Acquire::ForceIPv4=true update \
 && apt-get -o Acquire::ForceIPv4=true install -yq --no-install-recommends \
    apt-utils \
    build-essential \
    curl \
    xvfb \
    ffmpeg \
    xorg-dev \
    libsdl2-dev \
    swig \
    cmake \
    python-opengl \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

##################################### PIP3 #####################################

RUN pip3 install  \
    numpy \
    gym \
    box2d-py \
    torch \
    torchvision \
    matplotlib \
    seaborn \
    pandas

##################################### COPY #####################################

RUN mkdir ${HOME}/RLOpenAIGymGPU

COPY --chown=1000:1000 . ${HOME}/RLOpenAIGymGPU

################################## drlnd deps ##################################

WORKDIR ${HOME}/RLOpenAIGymGPU/python

RUN pip install .

#################################### CMAKE #####################################

# RUN mkdir ${HOME}/RLOpenAIGymGPU/build \
#  && cd ${HOME}/RLOpenAIGymGPU/build \
#  && cmake  .. \
#  && make -j2

##################################### TAIL #####################################

RUN chown ${NB_UID} ${HOME}/RLOpenAIGymGPU
 
USER ${NB_USER}

WORKDIR ${HOME}/RLOpenAIGymGPU
