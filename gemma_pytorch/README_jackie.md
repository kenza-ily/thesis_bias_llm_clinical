# sml-llm-bias
Code for 2024 MSc projects on bias in LLMs.

## Setup instructions

### Get Gemma

Follow the instructions at https://ai.google.dev/gemma/docs/setup to register for Kaggle.

Download Gemma 2B network weights from
https://www.kaggle.com/models/google/gemma/frameworks/pyTorch/variations/2b

Extract to a directory (such as `/tmp/ckpt`) and take note of that directory.

### Install Docker

Follow instructions to install docker engine.

```bash
sudo apt-get install docker-ce
sudo usermod -aG docker $USER
newgrp docker
```

### Build Gemma Docker image and test

```bash
DOCKER_URI=gemma:${USER}
docker build -f docker/Dockerfile ./ -t ${DOCKER_URI}

CKPT_DIR=<path to network weights dir>
PROMPT="The meaning of life is"

docker run -t --rm \
    -v ${CKPT_DIR}/gemma-2b.ckpt:/tmp/ckpt \
    ${DOCKER_URI} \
    python scripts/run.py \
    --ckpt=/tmp/ckpt \
    --variant="2b" \
    --prompt="${PROMPT}"
    # add `--quant` for the int8 quantized model.
```

You should see output formatted like:

```bash
Model loading done
======================================
PROMPT: The meaning of life is
RESULT: ...
======================================
```

You will always need to rebuild the Docker image (`docker build`) when modifying code.

## Style guide

Install the flake8 linter so that we can maintain a consistent and readable Python style in this repository. If you are using VSCode or another IDE-type editor, you can install it as a plugin.

On the command line you can install it e.g. using pip:

```bash
pip install flake8
```

Then use `flake8 <file to lint>` to get style suggestions.