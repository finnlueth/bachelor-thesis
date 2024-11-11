# syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install git-all -y

WORKDIR mnt/plmgen

RUN groupadd -g 30001 brennema && \
     useradd --system --create-home --uid 30001 --gid 30001 -s /bin/bash brennema

COPY --chown=brennema:brennema . .
RUN git config --global --add safe.directory /mnt/plmgen
RUN git remote set-url origin https://gitlab.lrz.de/0000000001496075/plmgen.git

RUN --mount=type=cache,target=/root/.cache/pip \
     pip cache info && \
     pip install -e .

USER brennema
RUN git config --global --add safe.directory /mnt/plmgen






