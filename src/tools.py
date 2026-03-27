import json
import re
import logging
from typing import List
import ast
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import tool

from utils import reconstruct_parent_context, expand_context, rerank_context, format_and_log_docs
from shared_resources import RESOURCES, AVAILABLE_COLLECTIONS

if RESOURCES is None:
    raise RuntimeError("共享资源未能成功初始化，请检查 shared_resources.py 的日志。")

HYBRID_RETRIEVER = RESOURCES["hybrid_retriever"]
FULL_KNOWLEDGE_BASE_DF = RESOURCES["full_knowledge_base_df"]
RERANKER_MODEL = RESOURCES["reranker_model"]
PARSER_LLM = RESOURCES["parser_llm"]
GENERATION_PROMPT = RESOURCES["generation_prompt"]
CYPHER_QA_CHAIN = RESOURCES["cypher_qa_chain"]

@tool
def retrieval_qa_tool(query: str) -> str:
    logging.info(f"======== TOOL: EXECUTING Retrieval_QA_Tool | Query: {query} ========")
    entities = [name for name in AVAILABLE_COLLECTIONS if name in query]
    rag_docs = HYBRID_RETRIEVER.retrieve(query, entities, is_global=(not entities))
    rag_docs = reconstruct_parent_context(rag_docs, FULL_KNOWLEDGE_BASE_DF)
    rag_docs = expand_context(rag_docs, FULL_KNOWLEDGE_BASE_DF)
    rag_docs = rerank_context(query, rag_docs, RERANKER_MODEL, final_k=5)
    formatted_context = format_and_log_docs(rag_docs)
    generation_chain = GENERATION_PROMPT | PARSER_LLM | StrOutputParser()
    final_answer = generation_chain.invoke({"context": formatted_context, "question": query})
    return final_answer.strip()
@tool
def analytical_query_tool(query: str) -> str:
    logging.info(f"======== TOOL: EXECUTING Analytical_Query_Tool | Query: {query} ========")
    if not CYPHER_QA_CHAIN:
        return "错误：图数据库查询链 (cypher_qa_chain) 未初始化。"
    result = CYPHER_QA_CHAIN.invoke({"query": query})
    final_answer = result.get('result', '未能从图数据库中生成最终答案。')
    return final_answer.strip()

DFA_SCORING_TABLE_MD_FOR_TOOL = (
    '| 评分项 | 得分 9 | 得分 3 | 得分 1 |\n'
    '|---|---|---|---|\n'
    '| 是否属于最少零件 | 该零件属于最少零件 | - | 该零件不属于最少零件 |\n'
    '| 可接受缺陷水平P | P <= 0.1% | 0.1% < P <= 1.5% | P > 1.5% |\n'
    '| 零件定向复杂度 | 零件送达时姿态正确，无需重新定向 | 零件姿态基本正确，但需简单旋转微调 | 零件姿态随机，需要复杂翻转或旋转 |\n'
    '| 零件坚固耐损度 | 零件坚固，不易碎也不易划伤 | 零件坚固但表面易划伤 | 零件由易碎材料制成 |\n'
    '| 零件缠结趋势 | 零件设计不易于相互缠结或搭扣 | - | 零件容易因形状（如钩子、弹簧）而相互缠结 |\n'
    '| 零件放置稳定性 | 零件重心低，有唯一的稳定静止面且即为正确的装配姿态 | 零件能稳定放置，但其稳定姿态并非正确的装配姿态 | 零件重心高或形状不规则，无稳定静止姿态 |\n'
    '| 零件的对称度 | α+β < 360° | 360° ≤ α+β < 540° | 540° ≤ α+β < 720° |\n'
    '| 零件的重量G | 0.1g <= G <= 2kg | 0.01g < G < 0.1g 或 2kg < G <= 6kg | G <= 0.01g 或 G > 6kg |\n'
    '| 零件最长边长度L | 5mm < L < 50mm | 2mm < L < 5mm或50mm < L < 200mm | L < 2mm或 L > 200mm |\n'
    '| 零件夹持便利性 | 零件具有可夹持面且可复用现有夹持器(对应ZH 90/120, ZH 210/240） | 零件具有可夹持面但需要专用夹持器（对应设备后缀带有“F”、“CR”、“HP”、“HV”后缀的或适配特定负载的型号如 "ZH 120", "ZH 160"） | 零件不具有可夹持面（对应9、3分都不符合） |\n'
    '| 装配动作 | 简单的直线按压或插入动作(对应设备后缀带有“PA”的机器人)  | 需要旋转的动作（如拧螺丝）(对应设备后缀不带有“PA”的机器人)  | 暂不使用 |\n'
    '| 装配操作可及性 | 臂展合适，无限制 (MaximumReach < 2400mm) | 臂展较长，可能影响精度 (2400mm <= MaximumReach <= 3000mm) | 臂展超长，操作笨重 (MaximumReach > 3000mm) |\n'
    '| 插入引导特征 | 零件带有倒角或圆角以辅助插入 | 无倒角，但有其他锥形或曲面特征辅助插入 | 无任何倒角或引导特征（直角插入） |\n'
    '| 公差范围T | T > 0.5 mm | 0.1 mm ≤ T ≤ 0.5 mm | T < 0.1 mm |\n'
    '| 装配后的自保持性 | 零件装入后立即自锁或稳固固定 | 零件能靠重力或凹槽保持位置，但未固定 | 装入后若不扶持，零件会掉落或移位 |\n'
    '| 紧固方法 | 无需紧固件:如卡扣、过盈配合(对应设备后缀带有“C”的机器人)) | 使用标准紧固件：如螺丝、铆钉(对应设备后缀不带有“C”和“F”的机器人) | 采用永久性连接：如焊接、胶合(对应设备后缀带有“F”或同时带有“C”和“F”的机器人)  |\n'
    '| 装配所需辅助设备 | 无需任何工具或辅助设备即可装配 | 需要简单的通用工具辅助（如螺丝刀、扳手） | 需要大型或专用的辅助设备（如压力机、焊接机器人） |\n'
    '| 检查/调整 | 无需检查零件是否到位 (对应6轴及以上) | 有必要检查零件是否到位或装配正确 (对应3-5轴) | 需要调整或重新定位零件 (对应3轴以下) |\n'
)


@tool
def dfa_scoring_tool(tool_input: str) -> str:
    logging.info(f"======== TOOL: EXECUTING DFA_Scoring_Tool (raw input: {tool_input}) ========")
    try:
        input_data = None
        if isinstance(tool_input, dict):
            input_data = tool_input
        elif isinstance(tool_input, str):
            cleaned_input = tool_input.strip().strip("'\"")
            try:
                input_data = json.loads(cleaned_input)
            except json.JSONDecodeError:
                try:
                    input_data = ast.literal_eval(cleaned_input)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"输入字符串既不是有效的JSON，也不是有效的Python字典: {e}")
        if input_data is None:
            raise ValueError("输入格式无法识别。")
        criterion_name = input_data.get("criterion_name")
        part_parameters = input_data.get("part_parameters")
        if not criterion_name or part_parameters is None:
            raise ValueError("解析后的输入缺少 'criterion_name' 或 'part_parameters' 键")
    except Exception as e:
        error_msg = f"输入格式错误: {e}"
        logging.error(error_msg, exc_info=True)
        return json.dumps({"score": "错误", "reasoning": error_msg, "value": "N/A"})

    logging.info(f"Criterion: {criterion_name} | Parameters: {part_parameters}")
    scoring_llm = RESOURCES.get("parser_llm")
    if not scoring_llm:
        return json.dumps({"score": "错误", "reasoning": "Scoring LLM未加载", "value": "N/A"})

    prompt_template = (
            '你是一个精确的DFA评分机器人。你的唯一任务是根据下面提供的DFA评分规则表，对一个给定的评分项进行打分。\n\n'
            '# DFA评分规则表:\n'
            + DFA_SCORING_TABLE_MD_FOR_TOOL + '\n\n'
                                              '# 你的任务:\n'
                                              '1. 仔细阅读用户提供的 **评分项名称** 和 **零件参数**。\n'
                                              '2. 在评分规则表中找到 **评分项名称** 对应的那一行。\n'
                                              '3. 从 **零件参数** 中提取出与该评分项相关的值。\n'
                                              '4. 根据该值，判断它属于哪一列（得分9, 3, 或 1）。\n'
                                              '5. 以JSON格式返回结果，必须包含四个字段: "score", "criterion", "value", "reasoning"。\n\n'
                                              '# 用户请求:\n'
                                              f'- 评分项名称: "{criterion_name}"\n'
                                              f'- 零件参数: "{part_parameters}"\n\n'
                                              '# 你的JSON输出 (只返回JSON代码块):\n'
    )
    response = scoring_llm.invoke(prompt_template)
    response_content = response.content if hasattr(response, 'content') else response
    match = re.search(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL)
    if match:
        return match.group(1)
    else:
        try:
            start = response_content.find('{')
            end = response_content.rfind('}')
            if start != -1 and end != -1:
                json_str = response_content[start:end + 1]
                json.loads(json_str)
                return json_str
        except json.JSONDecodeError:
            pass
        return json.dumps({"score": "错误", "reasoning": "评分工具未能生成有效的JSON。", "raw_output": response_content,
                           "value": "N/A"})


@tool
def suggestion_expert_tool(part_info: str, low_score_item_json: str) -> str:
    logging.info(f"======== TOOL: CALLING Suggestion_Expert_Tool ========")
    DFA_MODEL = RESOURCES.get("dfa_model")
    DFA_TOKENIZER = RESOURCES.get("dfa_tokenizer")
    if not DFA_MODEL or not DFA_TOKENIZER:
        return "错误：DFA专家模型 (dfa_model/dfa_tokenizer) 未加载。"

    try:
        item_data = json.loads(low_score_item_json)
        criterion = item_data.get('criterion', '未知项')
        score = item_data.get('score', '未知分')
        reasoning = item_data.get('reasoning', '无理由')
    except json.JSONDecodeError:
        return "错误：输入的评分项JSON格式不正确。"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
你是一位顶级的DFA（面向装配的设计）总工程师。你的核心职责是针对DFA评估中的低分项，提供清晰、具体、可执行的改进建议。你必须严格模仿下面的示例，基于【零件参数】和【评分标准】进行问题分析，然后给出具体的改进建议列表。
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
---
**这是一个必须模仿的示例:**

*   **[示例输入]**
    *   **评分项**: '零件的重量G'
    *   **零件参数**: '重量为10kg'
    *   **评分标准**: '得分9: G < 1kg | 得分3: 1kg <= G <= 5kg | 得分1: G > 5kg'
    *   **评估得分**: 1

*   **[你应该输出的格式]**
    **问题分析**: 该零件的重量为10kg，根据评分标准【G > 5kg 得1分】，这属于严重超重。过重的零件会大幅增加机器人在搬运和装配过程中的负载，导致能耗增加、运动速度降低，并可能需要选用更昂贵的大功率机器人。
    **改进建议**:
    1.  **材料替换**: 评估是否可以使用更轻的材料，例如将传统的钢材更换为7000系列铝合金或碳纤维增强复合材料，在保证强度的前提下大幅减重。
    2.  **结构优化**: 使用拓扑优化或有限元分析（FEA）软件，对零件进行力学分析，识别并去除所有非承载区域的冗余材料，实现轻量化设计。
    3.  **功能集成**: 审视该零件是否能与相邻的零件（如支架、外壳）进行一体化设计，通过减少零件总数和连接件（螺丝、卡扣）来从根本上降低系统总重量。
---

**现在，这是你的正式任务:**

*   **[任务输入]**
    *   **评分项**: '{criterion}'
    *   **零件情况 (包含参数和标准)**: "{part_info}"
    *   **评估得分**: {score}

请严格模仿上面的示例格式，对这个任务进行分析并给出1-3条具体的改进建议。
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    try:
        inputs = DFA_TOKENIZER(prompt, return_tensors="pt").to(DFA_MODEL.device)
        outputs = DFA_MODEL.generate(**inputs, max_new_tokens=1024, temperature=0.3, top_p=0.9,
                                     pad_token_id=DFA_TOKENIZER.eos_token_id)
        response = DFA_TOKENIZER.decode(outputs[0], skip_special_tokens=False)
        response_part = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        suggestion = response_part.split("<|eot_id|>")[0].strip()

        return suggestion
    except Exception as e:
        logging.error(f"执行 suggestion_expert_tool 时出错: {e}", exc_info=True)
        return f"生成建议时出错: {e}"