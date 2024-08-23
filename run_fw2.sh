#!/bin/bash -l
#$ -l mem=10G
#$ -l h_rt=20:00:00
#$ -j y
#$ -l gpu=2
#$ -N KB_FW2_exp2_GxE #! HYPERPARAMETERS
#$ -m be
#$ -M ucabkbe@ucl.ac.uk
#$ -cwd

echo "START"
module load python3/recommended
echo "Loaded Python"
# export PATH="$PATH:/home/ucabkbe/.ollama"

cd /home/ucabkbe/bias_llm_clinical_nle/

# Activate the environment
source /home/ucabkbe/bias_llm_clinical_nle/new_env_2/bin/activate
echo "Environment activated"
/home/ucabkbe/.ollama/ollama serve
# Start the server in the background
echo "Starting Ollama server..."
OLLAMA_HOST=127.0.0.1:12212 /home/ucabkbe/.ollama/ollama serve

# Wait for the server to start
sleep 10  # Adjust the sleep time if needed

# Pull models directly here
echo "Pulling models with Ollama..."
/home/ucabkbe/.ollama/ollama pull mistral:nemo
/home/ucabkbe/.ollama/ollama pull mixtral:8x22b
/home/ucabkbe/.ollama/ollama pull mistral:7b
/home/ucabkbe/.ollama/ollama pull llama3:8b
/home/ucabkbe/.ollama/ollama pull llama3:70b
/home/ucabkbe/.ollama/ollama pull llama3.1
/home/ucabkbe/.ollama/ollama pull gemma2:2b
/home/ucabkbe/.ollama/ollama pull gemma2

echo "Running the script"
# ! HYPERPARAMETERS
python scripts/run_exp2.py GxE 2 "test3_cluster" "open"
echo "Script running done"
echo "END"