import os
import logging
from typing import List, Dict
import json
import re
import torch
import pandas as pd
import numpy as np
import chromadb

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.chains.graph_qa.prompts import CYPHER_GENERATION_PROMPT
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_DIR = './KB/VectorDB'
COMBINED_CSV_PATH = './KB/RAG/narrative_combined_knowledge_base.csv'
MAPPING_FILE_PATH = "./KB/RAG/semantic_mapping.json"
EMBEDDING_MODEL_PATH = './Model/SFR-Embedding-Mistral'
RERANKER_MODEL_PATH = './Model/bge-reranker-v2-m3'
DFA_LLM_MODEL_PATH = "./Model/Output/Llama-3.1-8B-Instruct-Merge"
ROUTER_API_BASE_URL = "YOUR_URL_HERE"
ROUTER_API_KEY = "YOUR_API_KEY_HERE"
ROUTER_LLM_MODEL_NAME = "deepseek-v3.1"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "YOUR_PASSWORD_HERE"

AVAILABLE_COLLECTIONS = [
    "KUKA_KR_100_R3500_press", "KUKA_KR_100_R3500_pressC", "KUKA_KR_120_R1800_nano",
    "KUKA_KR_120_R1800_nanoC", "KUKA_KR_120_R2100_nanoFexclusive", "KUKA_KR_120_R2500_pro",
    "KUKA_KR_120_R2700_extraHA", "KUKA_KR_120_R2900_extra", "KUKA_KR_120_R2900_extra_C",
    "KUKA_KR_120_R2900_extra_C_F", "KUKA_KR_120_R2900_extra_C_F_HP", "KUKA_KR_120_R2900_extra_F_HP",
    "KUKA_KR_120_R3200_PA", "KUKA_KR_120_R3200_PAarctic", "KUKA_KR_120_R3500_press",
    "KUKA_KR_120_R3500_pressC", "KUKA_KR_120_R3500_primeK", "KUKA_KR_120_R3900_ultra_K",
    "KUKA_KR_120_R3900_ultraK_F", "KUKA_KR_120_R3900_ultraK_F_HP", "KUKA_KR_150_R2700_extra",
    "KUKA_KR_150_R2700_extra_C", "KUKA_KR_150_R2700_extra_C_F", "KUKA_KR_150_R2700_extra_C_F_HP",
    "KUKA_KR_150_R2700_extra_F", "KUKA_KR_150_R2700_extra_F_HP", "KUKA_KR_150_R3100_prime",
    "KUKA_KR_150_R3300_primeK", "KUKA_KR_150_R3700_ultraK", "KUKA_KR_150_R3700_ultraK_F",
    "KUKA_KR_150_R3700_ultraK_F_HP", "KUKA_KR_160_R1570_nano", "KUKA_KR_160_R1570_nanoC",
    "KUKA_KR_180_R2100_nano_F_exclusive", "KUKA_KR_180_R2500_extra", "KUKA_KR_180_R2500_extra_C",
    "KUKA_KR_180_R2500_extraC_F", "KUKA_KR_180_R2500_extra_F", "KUKA_KR_180_R2500_extraF_HP",
    "KUKA_KR_180_R2900_prime", "KUKA_KR_180_R3100_primeK", "KUKA_KR_180_R3200_PA",
    "KUKA_KR_180_R3200_PA_HO", "KUKA_KR_180_R3200_PAarctic", "KUKA_KR_180_R3500_ultra_K_F",
    "KUKA_KR_180_R3500_ultra_K_F_HP", "KUKA_KR_180_R3500_ultraK", "KUKA_KR_210_R2700_prime",
    "KUKA_KR_210_R2700_primeC", "KUKA_KR_210_R2700_primeCR", "KUKA_KR_210_R2700_primeF",
    "KUKA_KR_210_R2700_prime_C_F", "KUKA_KR_210_R2900_primeK", "KUKA_KR_210_R3100_ultra",
    "KUKA_KR_210_R3100_ultraC", "KUKA_KR_210_R3100_ultra_C_F", "KUKA_KR_210_R3100_ultra_F",
    "KUKA_KR_210_R3300_ultra_K_F", "KUKA_KR_240_R2500_prime", "KUKA_KR_240_R2900_ultra",
    "KUKA_KR_240_R2900_ultraC", "KUKA_KR_240_R2900_ultra_C_F", "KUKA_KR_240_R2900_ultra_F",
    "KUKA_KR_240_R3100_ultra_K_F", "KUKA_KR_240_R3200_PA", "KUKA_KR_240_R3200_PA_HO",
    "KUKA_KR_240_R3200_PAarctic", "KUKA_KR_270_R2700_ultra", "KUKA_KR_270_R2700_ultra_C",
    "KUKA_KR_270_R2700_ultraC_F", "KUKA_KR_270_R2700_ultraF", "KUKA_KR_270_R2900_ultra_K_F",
    "KUKA_KR_270_R2900_ultraK", "KUKA_KR_270_R3100_ultra_K_F", "KUKA_KR_270_R3100_ultraK",
    "KUKA_KR_300_R2500_ultra", "KUKA_KR_300_R2500_ultraC", "KUKA_KR_300_R2500_ultraC_F",
    "KUKA_KR_300_R2500_ultraF", "KUKA_KR_90_R2700_pro", "KUKA_KR_90_R2900_extra_HA",
    "KUKA_KR_90_R3100_extra", "KUKA_KR_90_R3100_extraC", "KUKA_KR_90_R3100_extraC_F",
    "KUKA_KR_90_R3100_extraC_F_HP", "KUKA_KR_90_R3100_extraF", "KUKA_KR_90_R3100_extraF_HP",
    "KUKA_KR_90_R3100_extraHA", "KUKA_KR_90_R3700_primeK",
]
class HybridRetriever:
    def __init__(self, embedding_model, client, bm25_indices, docs_by_model):
        self.embedding_model, self.client = embedding_model, client
        self.bm25_indices, self.docs_by_model = bm25_indices, docs_by_model
        self.available_collections = AVAILABLE_COLLECTIONS
    def retrieve(self, query: str, collections_to_search: List[str], k: int = 5, is_global: bool = False) -> List[Document]:
        if not collections_to_search or is_global:
            logging.info(f"执行全局(纯向量)检索...")
            docs = []
            for collection_name in self.available_collections:
                try:
                    collection = self.client.get_collection(name=collection_name)
                    results = collection.query(query_embeddings=[self.embedding_model.encode(query, normalize_embeddings=True).tolist()], n_results=3, include=["metadatas", "documents"])
                    if results and results.get("documents") and results["documents"][0]:
                        for i in range(len(results["documents"][0])):
                            docs.append(Document(page_content=results["documents"][0][i], metadata=results["metadatas"][0][i]))
                except Exception as e: logging.warning(f"全局检索集合 '{collection_name}' 时出错: {e}。")
            return docs
        else:
            logging.info(f"在 {collections_to_search} 库中进行定向(混合)检索...")
            retrieval_k = k * 4
            vector_results = self._vector_search(query, collections_to_search, retrieval_k)
            bm25_results = self._bm25_search(query, collections_to_search, retrieval_k)
            fused_docs = self._fuse_results(vector_results, bm25_results)
            return fused_docs[:k*2]
    def _vector_search(self, query: str, collections_to_search: List[str], k: int) -> List[Document]:
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        docs = []
        for collection_name in collections_to_search:
            try:
                collection = self.client.get_collection(name=collection_name)
                results = collection.query(query_embeddings=[query_embedding], n_results=k, include=["metadatas", "documents"])
                if results and results.get("documents") and results["documents"][0]:
                    for i in range(len(results["documents"][0])): docs.append(Document(page_content=results["documents"][0][i], metadata=results["metadatas"][0][i]))
            except Exception as e: logging.warning(f"Vector search on '{collection_name}' failed: {e}")
        return docs
    def _bm25_search(self, query: str, collections_to_search: List[str], k: int) -> List[Document]:
        tokenized_query = re.findall(r'\w+', query.lower())
        all_bm25_results = []
        for model_id in collections_to_search:
            if model_id in self.bm25_indices:
                bm25 = self.bm25_indices[model_id]
                corpus_docs = self.docs_by_model[model_id]
                doc_scores = bm25.get_scores(tokenized_query)
                top_n_indices = np.argsort(doc_scores)[::-1][:k]
                all_bm25_results.extend([corpus_docs[i] for i in top_n_indices])
        return all_bm25_results
    def _fuse_results(self, vector_docs: List[Document], bm25_docs: List[Document], k_const: int = 60) -> List[Document]:
        fused_scores, doc_map = {}, {doc.metadata.get('path'): doc for doc in vector_docs + bm25_docs if doc.metadata.get('path')}
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get('path')
            if doc_id:
                if doc_id not in fused_scores: fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1.0 / (k_const + rank + 1)
        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get('path')
            if doc_id:
                if doc_id not in fused_scores: fused_scores[doc_id] = 0
                fused_scores[doc_id] += 1.0 / (k_const + rank + 1)
        reranked_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in reranked_ids if doc_id in doc_map]



def create_parser_chain(llm, collection_names):
    parser_template = """你是一个专业的工业机器人领域查询路由专家。你的核心任务是分析用户问题，并将其精确地路由到两种核心能力之一，或将其分解为多步任务。

# 两种核心能力定义:

1.  **"Retrieval_QA" (检索式问答)**
    *   **职责**: 从【文本知识库】中查找**描述性、解释性、概念性**的信息。
    *   **适用场景**: 当用户想了解 "是什么"、"怎么样"、"为什么"、"介绍一下"、"对比优缺点"、"应用场景" 等**非精确数值**的问题时使用。
    *   **关键词**: 介绍、描述、优势、特点、应用、原理、配置要求。
    *   **示例**:
        *   "介绍一下 KR 120 R1800 nano C 的核心优势。" -> (应路由到 Retrieval_QA)
        *   "对比一下 nano 系列和 pro 系列的设计理念有什么不同？" -> (应路由到 Retrieval_QA)
        *   "这个机器人的安装有什么特别需要注意的地方？" -> (应路由到 Retrieval_QA)

2.  **"Analytical_Query" (分析式查询)**
    *   **职责**: 从【图数据库(KG)】中查询**精确的、结构化的技术参数**，并进行筛选、排序或计算。
    *   **适用场景**: 当用户想知道 "多少"、"哪个最..."、"列出所有..."、"筛选..." 等**涉及具体数值**的问题时使用。
    *   **关键词**: 具体参数、多少、最大/最小、最长/最短、重量、负载、臂展、速度、尺寸、列出、筛选、排序。
    *   **示例**:
        *   "KUKA KR 300 R2700-2 HC II 的最大负载是多少公斤？" -> (应路由到 Analytical_Query)
        *   "哪个机器人的重量最轻？" -> (应路由到 Analytical_Query)
        *   "列出所有额定负载超过200kg的机器人。" -> (应路由到 Analytical_Query)
        *   "KR 210 R2700-2 C-F 型号A3轴的运动范围是多少度？" -> (这是一个精确参数查询，应路由到 Analytical_Query)

3.  **"Task_Oriented" (任务型对话)**
    *   **职责**: 当一个问题无法通过单次查询解决，需要**结合两种能力**时，将其分解为多个步骤。
    *   **适用场景**: 推荐、选型、或者需要先查参数再做描述的复杂问题。
    *   **示例**:
        *   "帮我找一款负载和 KR 210 差不多，但更适合在狭小空间使用的机器人。"
        *   "推荐一款用于高精度点焊的机器人，并说明理由。"

# 输出格式规则 (与之前相同):
- 对于 "Retrieval_QA" 和 "Analytical_Query" 类型，输出:
  {{
    "query_type": "类型名",
    "entities": ["识别出的机器人型号"] or [],
    "refined_query": "重写后的查询语句"
  }}
- 对于 "Task_Oriented" 类型，输出 (每个步骤必须明确指定 `sub_query_type`):
  {{
    "query_type": "Task_Oriented",
    "tasks": [
      {{"step": 1, "sub_query_type": "Analytical_Query", "query": "查询KR 210的额定负载和占地面积"}},
      {{"step": 2, "sub_query_type": "Analytical_Query", "query": "筛选出所有额定负载在[Step 1 Result]范围内的机器人，并按占地面积从小到大排序"}},
      {{"step": 3, "sub_query_type": "Retrieval_QA", "query": "介绍一下[Step 2 Result]中排名第一的机器人的核心优势和应用场景"}}
    ]
  }}

# 实体提取规则 (与之前相同):
# - 你必须从下面的 "可用机器人型号实体" 列表中精确地、一字不差地提取实体。
# - 输出的实体字符串必须与列表中的格式完全一致。
# - 如果用户提到的型号不在列表中，则 "entities" 字段为空数组 []。

# 可用机器人型号实体 (必须严格匹配):
{collections}
---
# 用户问题:
"{question}"

# 你的JSON决策:"""
    prompt = ChatPromptTemplate.from_template(parser_template)
    formatted_collections = "\n".join([f"- {name}" for name in collection_names])
    chain = prompt.partial(collections=formatted_collections) | llm | JsonOutputParser()
    return chain.with_retry(stop_after_attempt=3)

def create_cypher_qa_chain(llm, graph: Neo4jGraph, mapping_file_path: str):
    system_hints = load_and_format_mapping(mapping_file_path)
    final_prompt_template = CYPHER_GENERATION_PROMPT.template + system_hints
    CYPHER_GENERATION_PROMPT_WITH_HINTS = PromptTemplate(
        input_variables=["schema", "question"],
        template=final_prompt_template
    )
    cypher_qa_chain = GraphCypherQAChain.from_llm(
        llm,
        graph=graph,
        cypher_prompt=CYPHER_GENERATION_PROMPT_WITH_HINTS,
        verbose=True,
        return_intermediate_steps=True,
        allow_dangerous_requests=True
    )
    return cypher_qa_chain

def load_and_format_mapping(mapping_file_path: str) -> str:
    try:
        with open(mapping_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if not content.strip():
                logging.error(f"映射文件 '{mapping_file_path}' 为空！")
                return ""
            mapping_data = json.loads(content)

        template = [
            "\n\n# 以下是系统级映射规则，你必须严格遵守:",
            "\n## 核心词汇表 (自然语言 -> 数据库字段):"
        ]
        for term, details in mapping_data.get("core_concepts", {}).items():
            original_id = details.get('original_id_short', 'N/A')
            node_type = details.get('node_type', 'N/A')
            template.append(
                f"# - 当提到 '{term}', 对应的 original_id_short 是 '{original_id}', 其节点类型是 {node_type}")

        template.append("\n## 查询规则:")
        for i, rule in enumerate(mapping_data.get("query_rules", []), 1):
            escaped_rule = rule.replace('{', '{{').replace('}', '}}')
            template.append(f"# {i}. {escaped_rule}")

        template.append("\n## 查询示例:")
        for example in mapping_data.get("few_shot_examples", []):
            escaped_question = example.get('question', '').replace('{', '{{').replace('}', '}}')
            escaped_cypher = example.get('cypher', '').replace('{', '{{').replace('}', '}}')
            template.append(f"# 用户问题: \"{escaped_question}\"")
            template.append(f"# 正确的Cypher: {escaped_cypher}")

        prompt_text = "\n".join(template)
        return prompt_text

    except json.JSONDecodeError as e:
        logging.error(f"加载或格式化映射文件 '{mapping_file_path}' 失败: 无效的JSON格式 - {e}")
        return ""
    except Exception as e:
        logging.error(f"加载或格式化映射文件 '{mapping_file_path}' 失败: {e}")
        return ""
def initialize_resources():
    logging.info("=" * 20 + " 正在初始化所有共享资源 " + "=" * 20)

    if not os.path.exists(COMBINED_CSV_PATH):
        logging.error(f"❌ 关键知识库文件未找到: {COMBINED_CSV_PATH}")
        return None
    if not os.path.isdir(DFA_LLM_MODEL_PATH):
        logging.error(f"❌ DFA-LLM 模型目录未找到: {DFA_LLM_MODEL_PATH}")
        return None

    graph = None
    try:
        graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
        logging.info("✅ 成功连接到Neo4j数据库。")
    except Exception as e:
        logging.error(f"❌ 连接Neo4j数据库失败: {e}。")

    client = chromadb.PersistentClient(path=DB_DIR)

    logging.info(f"✅ 正在加载知识库: {COMBINED_CSV_PATH}")
    full_knowledge_base_df = pd.read_csv(COMBINED_CSV_PATH)

    logging.info("为BM25准备数据...")
    docs_by_model = {}
    for _, row in full_knowledge_base_df.iterrows():
        try:
            text_content = str(row['text']) if pd.notna(row['text']) else ""
            meta = json.loads(row['metadata_json'])
            meta['path'] = row['path']
            doc = Document(page_content=text_content, metadata=meta)
            model_id = meta.get('model_id')
            if model_id in AVAILABLE_COLLECTIONS:
                if model_id not in docs_by_model: docs_by_model[model_id] = []
                docs_by_model[model_id].append(doc)
        except (json.JSONDecodeError, KeyError) as e:
            logging.warning(f"解析CSV文件行失败: {row}, 错误: {e}")
            continue

    logging.info("正在构建BM25索引...")
    bm25_indices = {}
    for model_id, model_docs in docs_by_model.items():
        if not model_docs: continue
        corpus = [doc.page_content for doc in model_docs]
        tokenized_corpus = [re.findall(r'\w+', str(doc).lower()) for doc in corpus]
        bm25_indices[model_id] = BM25Okapi(tokenized_corpus)
    logging.info(f"✅ BM25索引构建完成。")

    logging.info("🚀 正在加载所有模型...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device='cpu')
    reranker_model = CrossEncoder(RERANKER_MODEL_PATH, max_length=512, device='cpu')
    parser_llm = ChatOpenAI(model=ROUTER_LLM_MODEL_NAME, openai_api_base=ROUTER_API_BASE_URL,
                            openai_api_key=ROUTER_API_KEY, temperature=0.1)

    dfa_model, dfa_tokenizer = None, None
    try:
        logging.info(f"🚀 正在从本地路径加载 DFA 专家模型: {DFA_LLM_MODEL_PATH}...")
        dfa_tokenizer = AutoTokenizer.from_pretrained(DFA_LLM_MODEL_PATH)
        dfa_model = AutoModelForCausalLM.from_pretrained(
            DFA_LLM_MODEL_PATH,
            device_map="auto",
            torch_dtype=torch.float16
        )
        logging.info("✅ DFA 专家模型和 Tokenizer 加载成功。")
    except Exception as e:
        logging.error(f"❌ 加载 DFA 专家模型失败: {e}")

    hybrid_retriever = HybridRetriever(embedding_model, client, bm25_indices, docs_by_model)
    parser_chain = create_parser_chain(parser_llm, AVAILABLE_COLLECTIONS)
    cypher_qa_chain = create_cypher_qa_chain(parser_llm, graph, MAPPING_FILE_PATH) if graph else None

    generation_prompt = ChatPromptTemplate.from_template(
        "你是一个KUKA机器人专家助手。请仔细分析下面提供的上下文信息来回答问题。\nContext:\n{context}\nQuestion: {question}\nHelpful Answer (in Chinese):")

    logging.info("✅ 所有资源初始化完毕。")

    return {
        "graph": graph,
        "full_knowledge_base_df": full_knowledge_base_df,
        "hybrid_retriever": hybrid_retriever,
        "reranker_model": reranker_model,
        "parser_llm": parser_llm,
        "parser_chain": parser_chain,
        "cypher_qa_chain": cypher_qa_chain,
        "generation_prompt": generation_prompt,
        "dfa_model": dfa_model,
        "dfa_tokenizer": dfa_tokenizer
    }

RESOURCES = initialize_resources()