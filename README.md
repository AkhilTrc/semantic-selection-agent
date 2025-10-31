# Semantic Selection Agent

## Overview
Code Respository for the paper "Exploring and Improving the Innovation Capabilities of Large Language Models". This repository contains implementations for semantic selection models with a focus on the LLM-FLARE-SR model. It also provides data, plotting scripts, and statistical testing notebooks for model evaluation and comparison.

## Contents

- **data/**  
  Contains all datasets and relevant files used in the experiments and analysis. Please check this folder to explore the data.

- **graph_plot.py**  
  Script to regenerate graphical plots for each model. Run this script using the following command: `python3 semantic-selection-agent.graph_plot.py`


- **llm-rag.py**  
Main script to test the central LLM-FLARE-SR model. Execute it as follows: `python3 semantic-selection-agent.llm-rag.py`


- **stat_test.ipynb**  
A Jupyter Notebook containing cells for performing statistical tests and comparisons between different models. You can specify the model names inside this notebook to customize which models to compare.

## Usage Instructions

1. Review the datasets in the `data` folder to understand the input data.
2. Run `graph_plot.py` to regenerate evaluation plots for any of the models.
3. Use `llm-rag.py` to test and evaluate the core LLM-FLARE-SR model.
4. Open and run cells in `stat_test.ipynb` to conduct statistical significance tests and compare model results. Modify model names as needed within cells to tailor comparisons.

---

For further details, explore the individual files and scripts included in this repository.
