#!/bin/bash
#$ -l mem=32G
#$ -j y
#$ -l gpu=2
#$ -N KB_FW2_exp2_GxE
#$ -m be
#$ -M ucabkbe@ucl.ac.uk
#$ -cwd

# Load necessary modules and set up environment
module load python3/recommended
module load apptainer

# Set the path to the new virtual environment
ENV_OLLAMA_PATH=/lustre/home/ucabkbe/bias_llm_clinical_nle/env_ollama

cd /home/ucabkbe/bias_llm_clinical_nle/

# Start Ollama using Apptainer
export APPTAINER_BINDPATH=/scratch/scratch/$USER,/tmpdir,$ENV_OLLAMA_PATH:$ENV_OLLAMA_PATH
apptainer exec --bind $ENV_OLLAMA_PATH:$ENV_OLLAMA_PATH ollama.sif bash <<EOF
source $ENV_OLLAMA_PATH/bin/activate

# Verify Python version
python --version

ollama serve > out 2>&1 &
sleep 30

# Pull models
ollama pull mistral:nemo
ollama pull mistral:7b
ollama pull llama3:8b
ollama pull llama3.1
ollama pull gemma2:2b
ollama pull gemma2

# Run your Python script
python scripts/run_exp2.py GxE 2 "test5_cluster" "open"
EOF