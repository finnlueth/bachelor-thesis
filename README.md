# Bachelor Thesis by Finn Lueth

`Monorepo` for all things bachelor thesis.

## Overarching Idea

This bachelor thesis touches on various topics regarding Molecular Dynamics and drug discovery.

## Contains

* mdCATH to 3Di to PSSM → `./mdcath_to_3di`
* Protein JEPA → `./prot_jepa`
* Drug RAG → `./drug_rag`

Term: Winter 2024/25


## Git Submodules
```sh
git submodule update --init --recursive

git submodule add git@github.com:facebookresearch/jepa.git prot-jepa/src/submodules/jepa
git submodule add git@github.com:flagshippioneering/bio2token.git mdcath-to-3di/src/submodules/bio2token
git submodule add git@github.com:steineggerlab/foldseek.git mdcath-to-3di/src/submodules/foldseek
git submodule add git@github.com:A4Bio/FoldToken_open.git mdcath-to-3di/src/submodules/foldtoken
```



## (Average) Project Structure


```

```


## Links I want to remember

* [chroma generative model for designing proteins programmatically github](https://github.com/generatebio/chroma)
* [helixfold3 github](https://github.com/PaddlePaddle/PaddleHelix/tree/dev/apps/protein_folding/helixfold3)
* [MISATO Paper](https://www.nature.com/articles/s43588-024-00627-2#data-availability)
* [MISATO GitHub](https://github.com/t7morgen/misato-dataset?tab=readme-ov-file)
* [One-step nanoscale expansion microscopy reveals individual protein shapes](https://www.nature.com/articles/s41587-024-02431-9)
* [Beautiful Figures Video](https://www.youtube.com/watch?v=i-HAjex6VtM)


## Docker

```sh
# syntax=docker/dockerfile:1

docker buildx build --platform=linux/amd64 -t finn-image .docker/
docker container run -it -d --rm --env-file .docker/env.list -v $(pwd -P)/:/mnt/dev/ --platform linux/amd64 --name finn-container finn-image 
docker container exec -it finn-container "/bin/bash" 

docker container stop finn-container
docker container rm finn-container
docker image rm finn-image

docker builder prune
```
