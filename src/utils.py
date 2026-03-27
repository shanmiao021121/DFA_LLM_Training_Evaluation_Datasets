import logging
import json
from typing import List
import pandas as pd
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

def reconstruct_parent_context(initial_docs: List[Document], db_df: pd.DataFrame) -> List[Document]:
    final_docs_map = {doc.metadata.get('path'): doc for doc in initial_docs if doc.metadata.get('path')}
    logging.info("🧐 正在重建父节点上下文...")
    paths_to_check = [doc.metadata.get('path') for doc in initial_docs if doc.metadata.get('path')]

    for path in paths_to_check:
        parts = path.split('/')
        for i in range(len(parts) - 1, 0, -1):
            parent_path = '/'.join(parts[:i])
            if parent_path and parent_path not in final_docs_map:
                parent_row = db_df[db_df['path'] == parent_path]
                if not parent_row.empty:
                    row_data = parent_row.iloc[0]
                    new_meta = json.loads(row_data['metadata_json'])
                    new_meta['path'] = parent_path
                    final_docs_map[parent_path] = Document(page_content=row_data['text'], metadata=new_meta)

    reconstructed_docs = list(final_docs_map.values())
    logging.info(f"🧐 重建完成，文档数从 {len(initial_docs)} 增加到 {len(reconstructed_docs)}")
    return reconstructed_docs

def expand_context(initial_docs: List[Document], db_df: pd.DataFrame) -> List[Document]:
    final_docs_map = {doc.metadata.get('path'): doc for doc in initial_docs if doc.metadata.get('path')}
    logging.info("🌳 正在进行上下文扩展...")

    for doc in list(final_docs_map.values()):
        meta = doc.metadata
        path = meta.get("path")
        level = meta.get("level")

        if path and (level is not None and level < 3):
            child_df = db_df[db_df['path'].str.startswith(path + '/') & (db_df['path'] != path)]
            if not child_df.empty:
                for _, row in child_df.iterrows():
                    child_path = row['path']
                    if child_path not in final_docs_map:
                        child_meta = json.loads(row['metadata_json'])
                        child_meta['path'] = child_path
                        final_docs_map[child_path] = Document(page_content=row['text'], metadata=child_meta)

    expanded_docs = list(final_docs_map.values())
    logging.info(f"🌳 扩展完成，文档数从 {len(initial_docs)} 增加到 {len(expanded_docs)}")
    return expanded_docs

def rerank_context(query: str, docs: List[Document], reranker: CrossEncoder, final_k: int = 5) -> List[Document]:

    if not docs:
        return []

    logging.info(f"🚀 正在精排 {len(docs)} 个文档...")
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs, show_progress_bar=False)
    scored_docs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top_docs = [doc for score, doc in scored_docs[:final_k]]
    logging.info(f"🚀 精排完成，文档数精简至: {len(top_docs)}")
    return top_docs

def format_and_log_docs(docs: List[Document]) -> str:
    if not docs:
        logging.warning("最终上下文为空，将返回提示信息。")
        return "知识库中没有找到与问题直接相关的信息。"
    sorted_docs = sorted(docs, key=lambda d: d.metadata.get('path', ''))

    logging.info("\n" + "=" * 25 + " 📝 最终上下文 " + "=" * 25)
    context_parts = []
    for i, doc in enumerate(sorted_docs):
        path = doc.metadata.get('path', 'N/A')
        content = doc.page_content
        print(f"--- [上下文 {i + 1}] ---")
        print(f"  路径: {path}")
        print(f"  内容: {content}")

        context_parts.append(f"--- Path: {path} ---\n{content}")

    print("=" * (50 + 15) + "\n")

    return "\n\n".join(context_parts)

