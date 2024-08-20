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

# Activate the environment
source /home/ucabkbe/bias_llm_clinical_nle/new_env_2/bin/activate
echo "Environment activated"

cd /home/ucabkbe/bias_llm_clinical_nle/

# Start the server in the background
echo "Starting Ollama server..."
ollama serve > out 2>&1 &

# Wait for the server to start
sleep 10  # Adjust the sleep time if needed

# Pull models directly here
echo "Pulling models with Ollama..."
ollama pull mistral:nemo
ollama pull mixtral:8x22b
ollama pull mistral:7b
ollama pull llama3:8b
ollama pull llama3:70b
ollama pull llama3.1
ollama pull gemma2:2b
ollama pull gemma2

echo "Running the script"
# ! HYPERPARAMETERS
python scripts/run_exp2.py GxE 2 "test3_cluster" "open"
echo "Script running done"
echo "END"