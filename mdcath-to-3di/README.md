# MDcath to 3di

## Installation

### Micromamba

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

### Docker on Cluster

```sh
docker image pull nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

# --user 22367:22367
docker container run -it -d -v $(pwd -P)/:/mnt/code/ --cpus 8 --memory 32G --gpus 1 --env-file .docker/env.list --name finn-container nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

docker container exec -it --workdir /mnt/code/ -it finn-container "/bin/bash"

.docker/container_setup.sh

docker container stop finn-container

docker container rm finn-container
```

### Other Commands

```sh
ls -1 ./tmp/data/mdCATH/data | wc -l

du -h ./tmp/data/mdCATH/data

df . -h

ps -fA | grep python

nohup python ./src/scripts/download.py &
```
