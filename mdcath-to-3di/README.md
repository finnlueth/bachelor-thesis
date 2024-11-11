# MDcath to 3di

## Installation

### Docker

```sh
# Build the Docker image
docker build --tag finn . --no-cache -f ./envs/Dockerfile

# Run the Docker container
docker run -it \
    --cpus 8 \
    --memory 32G \
    --gpus 1 \
    --env-file ~/envs/env.list \
    --rm \
    finn
```