# MDcath to 3di

## Installation

### Conda

```sh
git submodule init
# or 
git submodule update --init

micromamba env create --file  ./envs/env_base.yml --prefix ./.venv -y
micromamba activate --prefix ./.venv
micromamba env remove --prefix ./.venv -y
micromamba deactivate

python envs/repo.py
```

### Docker

```sh
docker run -it --cpus 8 --memory 32G --gpus 1 --env-file ./envs/env.list --rm finn

docker run -it --cpus 8 --memory 100G --gpus 1 --env-file ~/.docker/env.list --detach --rm




docker build --tag finn-image . --file ./envs/Dockerfile

docker container run -t -d -v $(PWD)/:/mnt/code/ --name finn-container finn-image

docker container exec -it finn-container "/bin/bash && pip install -e /mnt/code/"

docker container exec -it finn-container "/bin/bash"

docker stop finn-container

docker image rm finn-image
```


```sh
docker build --tag finn-image . --file ./envs/Dockerfile --platform=linux/amd64
docker container run -t -d --rm -v $(PWD)/:/mnt/code/ --name finn-container finn-image

docker container run -t -d --rm \
-v $(PWD)/configs:/mnt/configs \
-v $(PWD)/entrypoints:/mnt/entrypoints \
-v $(PWD)/envs:/mnt/envs \
-v $(PWD)/notebooks:/mnt/notebooks \
-v $(PWD)/src:/mnt/src \
-v $(PWD)/tmp:/mnt/tmp \
--name finn-container finn-image
```


### Docker on Cluster

```sh
docker image pull nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.3

docker container run -it -d -v $(pwd -P)/:/mnt/code/ --name finn-container nvcr.io/nvidia/ai-workbench/python-cuda122:1.0.3  --cpus 9 --memory 16G --gpus 1 --env-file ~/.docker/env.list

docker container exec -it --workdir /mnt/code/ -it finn-container /bin/bash
```
