### Docker on Cluster

--mount=type=cache,target=/opt/conda/pkgs

```sh
docker buildx build --platform=linux/amd64 -t finn-image .docker/ -f .docker/Dockerfile.3
docker container run -it -d --rm --env-file .docker/env.list -v $(pwd -P)/:/home/lfi/mnt/dev/ --platform linux/amd64 --name finn-container finn-image 
docker container exec -it finn-container "/bin/bash" 

docker container stop finn-container
docker container rm finn-container
docker image rm finn-image

docker builder prune
```
