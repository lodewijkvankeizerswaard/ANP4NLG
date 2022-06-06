# Attentive Neural Processes for Natural Laguage Generation

## Getting started

1. Set up conda environment by running `conda env create --file environment_gpu.yaml`
2. Download dataset: `./prepare-wikitext-103.sh`
3. Preprocess dataset by converting it to binary format: `./preprocess.sh`
4. Train Neural Process language model: `./train.sh`

## Lisa Setup

1. Install Anaconda 
```
wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
chmod +x Anaconda3-2022.05-Linux-x86_64.sh
srun -t 60 ./Anaconda3-2022.05-Linux-x86_64.sh
```
2. Install conda environment `srun -t 60 conda env create -f environment_gpu.yml`
3. Download dataset `./prepare-wikitext-103.sh`
4. Activate environment `conda activate anp4nlg_gpu`
5. Preprocess dataset by converting it to binary format: `./preprocess.sh`
