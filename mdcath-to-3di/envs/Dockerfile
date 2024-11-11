# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install git-all curl -y

if [ ! -d "/mnt/.venv" ]; then
RUN if [ ! -d "/mnt/.venv" ] ; then echo Argument not provided ; else echo Argument is $arg ; fi
# ENV MAMBA_ROOT_PREFIX=/opt/micromamba
# ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH
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




