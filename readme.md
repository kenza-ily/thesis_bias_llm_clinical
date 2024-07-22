# Evaluating bias in challenging medical question answering with LLMs

## Project Overview

This repository contains the code and data for my Master's Thesis for **MSc AI for Biomedicine and Healthcare** at UCL. I'm investigating bias mitigation in Large Language Models (LLMs) within clinical contexts. The project focuses on leveraging explainability techniques to reduce bias in LLMs applied to complex clinical cases. Key aspects of the research include:

1. Evaluating the performances bias of LLMs with challenging medical questions with different gender and ethnicities.
2. Exploring the application of Natural Language Explanation (NLE) techniques for bias mitigation in clinical LLMs.
3. Evaluating LLM performance on complex, real-world clinical cases using the [JAMA Network Clinical Challenge archive](https://jamanetwork.com/collections/44038/clinical-challenge)

This research builds upon the UNESCO 2024 Report on Gender Bias in LLM and aims to develop a comprehensive and rigorous framework for mitigating bias in clinical LLMs, ultimately promoting fairness and reducing the potential for discriminatory outputs in healthcare AI applications.

## Table of Contents

- [Evaluating bias in challenging medical question answering with LLMs](#evaluating-bias-in-challenging-medical-question-answering-with-llms)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Project Structure](#project-structure)
  - [Setup](#setup)
  - [Running Experiments](#running-experiments)
  - [Data](#data)
  - [Results](#results)
  - [Contributing](#contributing)

## Project Structure

```text
bias_llm_clinical_challenge/
│
├── config/                 # Configuration files
├── data/                   # Data files
│   ├── augmented/          # Augmented and filtered datasets
│   ├── processed/          # Processed data
│   └── raw/                # Raw data files
├── llm/                    # LLM-related code
├── notebooks/              # Jupyter notebooks for analysis
├── prompts/                # Prompt files for experiments
├── results/                # Experiment results
│   ├── experiment1/        # Results for different experiment1 variations
│   └── tests/              # Test results and scripts
├── scripts/                # Utility scripts
├── .gitignore
├── keys.env                # Environment variables (not tracked by git)
├── main.py                 # Main script to run all experiments
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/kenzaiily/bias_llm_clinical_challenge.git
   cd bias_llm_clinical_challenge
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `keys.env.example` to `keys.env`
   - Fill in your API keys and other sensitive information in `keys.env`

## Running Experiments

1. Run all tests:

   ```bash
   python results/tests/run.py
   ```

2. Run Experiment 1 with gender dataset:

   ```bash
   python scripts/run_experiment1.py jama_gender_filtered
   ```

3. Run Experiment 1 with gender x ethnicity dataset:

   ```bash
   python scripts/run_experiment1.py jama_genderxethniciy
   ```

4. Run all experiments:

   ```bash
   python main.py
   ```

## Data

- `data/augmented/jama_gender_filtered.csv`: Dataset filtered for gender analysis
- `data/augmented/jama_genderxethniciy.csv`: Dataset for combined gender and ethnicity analysis

Data preprocessing steps can be found in `data/eda.ipynb`.

## Results

Experiment results are stored in the `results/` directory. Each experiment variation has its own subdirectory with:

- `data.csv`: Input data for the experiment
- `results.csv`: Output data from the experiment
- `analysis.ipynb`: Jupyter notebook for analyzing the results

## Contributing

If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add some feature'`)
5. Push to the branch (`git push origin feature/your-feature-name`)
6. Create a new Pull Request

---

For any questions or issues, please open an issue on the GitHub repository.