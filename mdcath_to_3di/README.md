# MDcath to 3di

## Installation

### Conda

```sh
micromamba env create --file environment.yml --prefix ./.venv -y
micromamba activate --prefix ./.venv
pip install -e .

micromamba env remove --prefix ./.venv -y
micromamba deactivate

```
