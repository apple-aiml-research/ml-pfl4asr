FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 as base
RUN groupadd -g 1234 pflgroup && useradd -m -u 1234 -g pflgroup pfluser
# Switch to the custom user
USER pfluser
# Set the workdir
WORKDIR /home/pfluser

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/libcuda.so.1

# handle tzdata (emacs-nox dependency)
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles \
    CUDA_HOME=/usr/local/cuda-12.3

# Useful packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    curl \
    git \
    debianutils \
    openssh-client \
    openssh-server \
    openssl \
    util-linux \
    ldap-utils \
    findutils \
    htop \
    screen \
    iputils-ping \
    locales \
    lsof \
    ncdu \
    netcat \
    rsync \
    sysstat \
    tcpdump \
    traceroute \
    vim \
    tmux \
    wget \
    gnupg \
    tzdata \
    emacs-nox \
    mlocate \
    cmake \
    clang-format \
    # nvtop
    libudev-dev \
    libsystemd-dev \
    libdrm-dev \
    libncurses-dev \
    # CUDA req
    libhwloc-dev \
    # python3.10
    python3 \
    python3-pip \
    python3-dev \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1 \
    && rm -rf /var/lib/apt/lists/* \
    # Make sure we have this particular locale since it is convenient
    && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
