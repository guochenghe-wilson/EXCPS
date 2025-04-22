# Project Structure and Usage Instructions

This repository includes scripts and data directories for evaluating the performance of a proposed framework using various baselines and metrics. Below is a description of the key components and their functions:

## Data Directories

- **`output_jsons_after_logical_calculation/`**  
  Contains JSON files that store the results after logical inference is performed on the input data. This directory reflects the intermediate outcomes of the reasoning process within the proposed framework.

- **`output_jsons_after_timeshap/`**  
  Stores the JSON files generated after key feature identification is applied. This stage typically involves interpretability techniques to extract influential features.

- **`processed_data_new/`**  
  Includes the raw data collected from the real-world scene. This dataset serves as the initial input for all subsequent computational processes.

## Script Descriptions

- **`LLM_accuracy_calculation.py`**  
  Executes the computation of the BERTScore metric for evaluating the proposed framework. The script compares generated responses to reference answers to assess semantic similarity.

- **`bert_baseline1.ipynb`**  
  Implements a baseline evaluation using Retrieval-Augmented Generation (RAG) combined with key feature identification. The notebook calculates the corresponding BERTScore for comparison.

- **`bert_baseline2.ipynb`**  
  Implements a baseline evaluation using RAG without incorporating key feature identification. BERTScore is computed to benchmark performance against other methods.

- **`logic_accuracy_calculation.py`**  
  Computes numerical metrics to evaluate the logical inference capability of the proposed framework. This includes accuracy measures tailored to assess structured reasoning outputs.
