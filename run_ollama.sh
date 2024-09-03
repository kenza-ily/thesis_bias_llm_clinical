#!/bin/bash
#$ -l mem=32G
#$ -l h_rt=72:00:00
#$ -j y
#$ -l gpu=2
#$ -N KB_FW2_exp2_GxE
#$ -m be
#$ -M ucabkbe@ucl.ac.uk
#$ -cwd

# Load necessary modules and set up environment
module load python3/recommended
module load apptainer

cd /home/ucabkbe/bias_llm_clinical_nle/
source new_env_2/bin/activate

# Start Ollama using Apptainer
export APPTAINER_BINDPATH=/scratch/scratch/$USER,/tmpdir
apptainer shell ollama.sif <<EOF
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
python scripts/run_exp2.py GxE 2 "test3_cluster" "open"
EOF