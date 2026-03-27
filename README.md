# LLM Agent-driven Automated DFA Assessment with AAS-based RAG

This is the official repository for the paper: **"LLM Agent-driven Automated DFA Assessment with Fine-tuning and AAS-based RAG"**. 

We propose an industrial-grade Design for Assembly (DFA) evaluation framework based on a hierarchical **"Controller-Agent-Tool"** architecture, integrating domain-specific fine-tuning with an Asset Administration Shell (AAS)-aware hybrid RAG system.

## 📌 Core Contributions

1.  **DFA-Bench (Domain Knowledge Benchmark)**: The first public benchmark for automated DFA evaluation, containing **370 expert-verified items** across 18 dimensions (e.g., symmetry, weight, accessibility).
2.  **AAS-QA (Retrieval Accuracy Benchmark)**: A specialized dataset for industrial RAG performance validation. It consists of **100 query items** (50 semantic, 50 factual/structural) covering single-hop and multi-hop reasoning. Our system achieves a **91.4% exact match rate** on factual queries.
3.  **Controller-Agent-Tool Framework**: A hierarchical architecture that reconciles procedural rigor (Deterministic Controller) with cognitive flexibility (Generative Agent).
4.  **AAS-to-RAG Integration**: A domain-specific strategy mapping the **Asset Administration Shell (AAS)** standard to a dual-path RAG pipeline (Vector + Knowledge Graph), ensuring a "Single Source of Truth" for heterogeneous industrial data.

## 📁 Repository Structure

```text
DFA-Agent-Framework/
├── src/                        # Source Code
│   ├── app.py                  # Main Controller & Web UI 
│   ├── agent_main.py           # CLI-based Agent Orchestrator
│   ├── tools.py                # Definition of the Agent Toolset
│   ├── utils.py                # RAG Algorithms 
│   └── shared_resources.py     # Resource initialization 
├── data/                       # Datasets & Knowledge Base
│   ├── benchmarks/
│   │   ├── DFA-Bench.json      # 370-item domain knowledge benchmark
│   │   ├── AAS-QA-Semantic.json # 50 semantic query items for RAG testing
│   │   └── AAS-QA-Factual.jsonl # 50 factual query items for KG testing
│   ├── knowledge_base/
│   │   ├── narrative_kb_sample.csv # Sample AAS-modeled industrial knowledge
│   │   └── semantic_mapping.json   # Logic for Natural Language to KG (Cypher) mapping
│   ├── logic/
│   │   └── DFA_SCORING_TABLE.jsonl # Standardized 18-dimension scoring criteria
│   └── evaluation/
│       └── Agent_Test_Cases.json   # 30 industrial scenarios for Agent stress testing
└── README.md                   # Documentation
/

## 🏗️ System Architecture & Prompt Logic

### 1. Scoring Agent Reasoning Paradigm
The system utilizes the **ReAct (Reasoning and Acting)** paradigm to eliminate hallucinations in high-stakes engineering tasks:
*   **Thought**: Identifies specific parameters required for a criterion (e.g., alpha/beta angles for symmetry).
*   **Action**: Dynamically invokes `analytical_query_tool` (KG/Cypher) or `retrieval_qa_tool` (Vector/BM25).
*   **Observation**: Retrieves grounded physical parameters from the AAS knowledge base.
*   **Final Answer**: Generates a structured JSON report via rule-based logic.

### 2. Information Purity Principle
To mitigate "contextual noise" common in long-context LLMs, we enforce **Information Purity** at the prompt level. The Agent is constrained to pass *only* the atomic parameters directly relevant to the current scoring criterion into the `dfa_scoring_tool`.

## 📊 Datasets

The datasets used for the benchmarks in the paper are available in the `./data` directory:
*   **`DFA-Bench.json`**: [370 items] Expert-verified questions for evaluating domain reasoning and scoring logic.
*   **`AAS-QA.json`**: [100 items] For evaluating retrieval precision and structural fact-finding.
*   **`DFA-SFT-Sample.json`**: [6,000 samples] The Supervised Fine-tuning (SFT) dataset used to align the base model with DFA optimization tasks, including:
    *   *Reverse QA*: High-level improvement suggestions.
    *   *Knowledge-Point Extraction*: Fine-grained technical principle matching.

## 🛠️ The Industrial Toolset
## Tool Name	## Underlying Logic	## Capability
dfa_scoring_tool	Rule-based mapping of 18 DFA criteria	Deterministic Evaluation
retrieval_qa_tool	Vector Search + AAS Context Reconstruction + Rerank	Semantic Understanding
analytical_query_tool	Schema Injection + NL-to-Cypher	Precise Fact Retrieval
suggestion_expert_tool	Fine-tuned DFA-LLM (Llama-3.1-8B based)	Root Cause Analysis

## License

The contents of this repository are released under the [MIT License](https://opensource.org/licenses/MIT).

## Contact

For any questions or collaborations, please feel free to contact:

*   Jiaxin Liu - liujiaxin200211@163.com
*   https://github.com/shanmiao021121

