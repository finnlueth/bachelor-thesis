### Docker on Cluster

```sh
docker image pull nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

# --user 22367:22367
docker container run -it -d -v $(pwd -P)/:/mnt/code/ --cpus 8 --memory 32G --gpus 1 --env-file .docker/env.list --name finn-container nvcr.io/nvidia/ai-workbench/python-cuda120:1.0.3

docker container exec -it --workdir /mnt/code/ -it finn-container "/bin/bash"

.docker/container_setup.sh

docker container stop finn-container

docker container rm finn-container