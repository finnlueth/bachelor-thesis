### Docker on Cluster

--mount=type=cache,target=/opt/conda/pkgs

```sh
docker buildx build --platform=linux/amd64 -t finn-image .docker/ -f .docker/Dockerfile.3
```

#### Cluster

```sh
docker container run -it --cpus 8 --memory 32G --gpus 1 -d --rm --env-file .docker/env.list -v $(pwd -P)/:/home/lfi/mnt/dev/ --name finn-container finn-image 
docker container exec -it finn-container "/bin/bash" 

docker container stop finn-container
```

#### Local

```sh
docker container run -it -d --rm --env-file .docker/env.list -v $(pwd -P)/:/home/lfi/mnt/dev/ --name finn-cuda-container nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3
docker container exec -it finn-cuda-container "/bin/bash" 

docker container stop finn-cuda-container
docker container rm finn-cuda-container


docker container run -it -d --rm --env-file .docker/env.list -v $(pwd -P)/:/home/lfi/mnt/dev/ --platform linux/amd64 --name finn-container mambaorg/micromamba:git-9320035-cuda12.1.1-ubuntu20.04
docker container exec -it finn-container "/bin/bash" 

docker container stop finn-container
docker container rm finn-container

docker image rm finn-image

docker builder prune
```
