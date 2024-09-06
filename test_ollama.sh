#!/bin/bash
#$ -l mem=4G
#$ -l h_rt=00:10:00
#$ -j y
#$ -N test_ollama_env_2
#$ -m be
#$ -M ucabkbe@ucl.ac.uk
#$ -cwd

# Load necessary modules
. /etc/profile.d/modules.sh
module load python3/recommended
module load apptainer

# Set and print the path to the new virtual environment
ENV_OLLAMA_NEW_PATH=/lustre/home/ucabkbe/bias_llm_clinical_nle/env_ollama_new_copy
echo "ENV_OLLAMA_NEW_PATH: $ENV_OLLAMA_NEW_PATH"

# Print current directory and list its contents
echo "Current directory: $(pwd)"
ls -l

# Check Python executable outside container
echo "Python executable outside container:"
ls -l $ENV_OLLAMA_NEW_PATH/bin/python
echo "Actual Python executable:"
ls -l $ENV_OLLAMA_NEW_PATH/bin/python3
file $ENV_OLLAMA_NEW_PATH/bin/python3

# Verify Apptainer binding
echo "Verifying Apptainer binding:"
apptainer exec --bind $ENV_OLLAMA_NEW_PATH:$ENV_OLLAMA_NEW_PATH ollama.sif ls -l $ENV_OLLAMA_NEW_PATH/bin

# Start Apptainer
export APPTAINER_BINDPATH=/lustre
apptainer exec --bind $ENV_OLLAMA_NEW_PATH:$ENV_OLLAMA_NEW_PATH ollama.sif bash -c "
echo 'Inside Apptainer container'
export PATH=$ENV_OLLAMA_NEW_PATH/bin:\$PATH
export PYTHONPATH=$ENV_OLLAMA_NEW_PATH/lib/python3.9/site-packages:\$PYTHONPATH
echo \"PATH inside container: \$PATH\"
echo \"PYTHONPATH inside container: \$PYTHONPATH\"
which python3
python3 --version
$ENV_OLLAMA_NEW_PATH/bin/python3 test_env.py
"

echo "Job finished"