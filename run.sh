#!/bin/bash -l
#$ -l mem=1G
#$ -l h_rt=20:00:00
#$ -j y
#$ -l gpu=2
#$ -N run_emotion_experiments
#$ -m be
#$ -M ucabkbe@ucl.ac.uk
#$ -cwd
echo "Running python"
module load python3/recommended
echo "loaded python"
# Activate the environment
source /home/ucabkbe/Scratch/testenv/bin/activate
echo "activated the environment"
ollama serve > out 2>&1 &
echo "activated ollama"
# pull model directly here
python /home/ucabkbe/Scratch/msc_bias_llm_project/emotion_experiments/code/run_persona_emotion_bias.py
echo "ran emotion"
echo "finished running the code"




export OLLAMA_PORT=11435
export OLLAMA_MODELS=~/Scratch/ollama
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT

echo $OLLAMA_MODELS
echo $OLLAMA_HOST

ollama serve


ollama pull llama3.1


