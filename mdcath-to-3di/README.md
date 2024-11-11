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
