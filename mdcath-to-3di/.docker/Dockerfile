# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install git-all curl -y && apt -y upgrade
RUN apt install wget build-essential software-properties-common libssl-dev libffi-dev python3-dev libgdbm-dev libc6-dev libbz2-dev libsqlite3-dev tk-dev libffi-dev zlib1g-dev -y
RUN python -m pip install --upgrade pip

RUN wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz \
    && tar -xvf Python-3.12.2.tgz \
    && cd Python-3.12.2/ \
    && ./configure --enable-optimizations \
    && make -j 6 \
    && make altinstall

RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
    && ./bin/micromamba shell init -s bash -r ~/micromamba

RUN groupadd -g 22367 lfi \
    && useradd --system --create-home --uid 22367 --gid 22367 -s /bin/bash lfi

WORKDIR /mnt/code/
RUN micromamba activate \
    && micromamba env create --file .docker/env_docker.yml -y \
    && micromamba clean --all --yes

RUN echo "micromamba activate dev" >> ~/.bashrc \
    && echo "cd /mnt/code" >> ~/.bashrc \
    && echo "su lfi" >> ~/.bashrc

# COPY ./envs/env_docker.yml /mnt/envs/env_docker.yml
# RUN micromamba config set extract_threads 1 
# RUN micromamba env create --file /mnt/envs/env_docker.yml --prefix /mnt/.venv -y
# # RUN rm /mnt/envs/env_docker.yml


# ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
# RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
#     && mv bin/micromamba /usr/local/bin/ \
#     && rm -rf bin \
#     && micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX \
#     && micromamba create -f /mnt/env_cuda.yml



# RUN if [ ! -d "/mnt/.venv" ] ;\
# then echo .venv found ;\
# else echo .venv not found ; fi



# RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba \
#     && mv bin/micromamba /usr/local/bin/ \
#     && rm -rf bin \
#     && micromamba shell init -s bash -p $MAMBA_ROOT_PREFIX

# WORKDIR /mnt/

# # Copy the environment file
# COPY envs/env_cuda.yml /mnt/envs/env_cuda.yml

# # Add entrypoint script
# COPY entrypoint.sh /usr/local/bin/entrypoint.sh
# RUN chmod +x /usr/local/bin/entrypoint.sh

# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]




