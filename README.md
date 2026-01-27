# LLM Agent-driven Automated DFA Assessment Datasets

This repository hosts the open-source datasets and evaluation benchmarks developed for the research presented in our paper: **"LLM Agent-driven Automated DFA Assessment with Fine-tuning and AAS-based RAG"**.

## 1. Introduction

Our work introduces an automated Design-for-Assembly (DFA) evaluation framework leveraging LLM Agents, domain-specific fine-tuning, and a hybrid Retrieval-Augmented Generation (RAG) system. To ensure the transparency, reproducibility, and continuous improvement of our research, we are open-sourcing the following key datasets:

*   **DFA Supervised Fine-tuning (SFT) Dataset**: Used to align our DFA-LLM expert model with task-specific optimization suggestion generation.
*   **DFA-Bench Evaluation Benchmark**: The first public comprehensive evaluation dataset for DFA, used to rigorously assess our model's performance.

## 2. Dataset Descriptions

### 2.1. DFA Supervised Fine-tuning (SFT) Dataset

This dataset comprises a **6,000-sample instruction set** specifically designed for supervised fine-tuning (SFT) of our DFA-LLM expert model, as detailed in Section 3.2.1 (2) of our paper. It enables the model to generate specific, engineering-feasible optimization suggestions.

*   **Content**:
    *   **Strategy 1 (Reverse Question-Answer Generation)**: Approximately 2,000 high-quality samples.
    *   **Strategy 2 (Knowledge Point Extraction and Augmentation)**: Approximately 4,000 additional samples, derived from golden answers and synthesized into diverse instruction formats.
*   **Format**: The SFT dataset is provided in `DFA-SFT.json` (JSONL format), where each line is a JSON object representing an instruction-output pair.
*   **Usage**: This dataset can be used to fine-tune large language models for domain-specific instruction following and optimization suggestion generation in DFA-related tasks.

### 2.2. DFA-Bench Evaluation Benchmark

DFA-Bench is the **first public comprehensive evaluation dataset for Design for Assembly (DFA)**, constructed to objectively assess DFA evaluation models (Section 4.1.2 (1) of our paper).

*   **Content**: It contains **370 questions** designed by domain experts, covering fundamental DFA concepts, design principles, and application scenarios. The questions are categorized into:
    *   Single-choice questions
    *   Multiple-choice questions
    *   True/false questions
*   **Format**: The DFA-Bench evaluation benchmark is provided in `DFA-Bench.json` (JSONL format), where each line represents a question with its options and correct answer.
*   **Usage**: Researchers can use DFA-Bench to benchmark the accuracy and capabilities of their DFA evaluation models in a standardized and reproducible manner.

## 3. How to Access and Use

### 3.1. Downloading the Datasets

You can clone this repository to your local machine using Git:

```bash
git clone https://github.com/shanmiao021121/DFA_LLM_Training_Evaluation_Datasets.git

### 3.2. Data Structure

The datasets are directly available in the root directory of this repository:
DFA_LLM_Training_Evaluation_Datasets/
├── DFA-SFT.json         # DFA Supervised Fine-tuning Dataset
├── DFA-Bench.json       # DFA-Bench Evaluation Benchmark
├── LICENSE              # MIT License
└── README.md            # This README file

## 4. Citation

If you use this dataset in your research, please cite our forthcoming paper:
"LLM Agent-driven Automated DFA Assessment with Fine-tuning and AAS-based RAG" by Hanyu Jiang, Jiacheng Zhong, and Cheng Wang.
Full citation details (BibTeX) will be provided here once the paper is published or a preprint is available.

## 5. License

The contents of this repository are released under the [MIT License](https://opensource.org/licenses/MIT).

## 6. Contact

For any questions or collaborations, please feel free to contact:

*   Jiaxin Liu - liujiaxin200211@163.com
*   https://github.com/shanmiao021121

