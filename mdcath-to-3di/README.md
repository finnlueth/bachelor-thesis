# MDcath to 3di

## Installation

### Micromamba

```sh
git submodule update --init --recursive
maturin develop

micromamba env create --file  ./envs/env.yml --prefix ./.venv -y
micromamba activate --prefix ./.venv
micromamba deactivate
micromamba env remove --prefix ./.venv -y

python envs/repo.py
```

### Other Commands

```sh
ls -1 ./tmp/data/mdCATH/data | wc -l

du -h ./tmp/data/mdCATH/data

df . -h

ps -fA | grep python

nohup python ./src/scripts/download.py &
```
