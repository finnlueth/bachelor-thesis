export DEBIAN_FRONTEND=noninteractive
apt-get update -y && apt install git-all curl -y
apt -y upgrade
python -m pip install --upgrade pip
apt install build-essential software-properties-common libssl-dev libffi-dev python3-dev libgdbm-dev libc6-dev libbz2-dev libsqlite3-dev tk-dev libffi-dev zlib1g-dev -y
apt install wget -y
cd ~
wget https://www.python.org/ftp/python/3.12.2/Python-3.12.2.tgz
tar -xvf Python-3.12.2.tgz
cd Python-3.12.2/
./configure --enable-optimizations
make -j 6
make altinstall
echo "alias python=python3.12" >> ~/.bashrc
rm -rf Python-3.12.2
rm Python-3.12.2.tgz
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
./bin/micromamba shell init -s bash -r ~/micromamba
source ~/.bashrc
cd /mnt/code
micromamba env create --file  .docker/env_docker.yml --prefix ./.venv -y