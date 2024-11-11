# MDcath to 3di

## Installation

### Conda

```sh
git submodule init
# or 
git submodule update --init

micromamba env create --file  ./env/environment.yml --prefix ./.venv -y
micromamba activate --prefix ./.venv
micromamba env remove --prefix ./.venv -y
micromamba deactivate

python envs/repo.py
```

### Docker

```sh
docker build --tag finn . --no-cache -f ./envs/Dockerfile

docker run -it --cpus 8 --memory 32G --gpus 1 --env-file ./envs/env.list --rm finn


docker run -it --cpus 8 --memory 100G --gpus 1 --env-file ~/.docker/env.list --detach --rm
```
