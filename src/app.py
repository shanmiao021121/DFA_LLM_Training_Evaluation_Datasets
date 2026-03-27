import gradio as gr
import logging
import json
import io
import contextlib
import time
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# 导入您的项目模块
from tools import dfa_scoring_tool, retrieval_qa_tool, analytical_query_tool, suggestion_expert_tool
from shared_resources import RESOURCES, ROUTER_API_BASE_URL, ROUTER_API_KEY, ROUTER_LLM_MODEL_NAME

DFA_SCORING_TABLE = {
    "是否属于最少零件": "得分9: 该零件属于最少零件 | 得分3: - | 得分1: 该零件不属于最少零件 ",
    "可接受缺陷水平P": "得分9: P <= 0.1% | 得分3: 0.1% < P <= 1.5% | 得分1: P > 1.5%",
    "零件定向复杂度": "得分9: 零件送达时姿态正确，无需重新定向 | 得分3: 零件姿态基本正确，但需简单旋转微调 | 得分1: 零件姿态随机，需要复杂翻转或旋转",
    "零件坚固耐损度": "得分9: 零件坚固，不易碎也不易划伤 | 得分3: 零件坚固但表面易划伤 | 得分1: 零件由易碎材料制成",
    "零件缠结趋势": "得分9: 零件设计不易于相互缠结或搭扣 | 得分3: - | 得分1: 零件容易因形状（如钩子、弹簧）而相互缠结",
    "零件放置稳定性": "得分9: 零件重心低，有唯一的稳定静止面且即为正确的装配姿态 | 得分3: 零件能稳定放置，但其稳定姿态并非正确的装配姿态 | 得分1: 零件重心高或形状不规则，无稳定静止姿态，容易滚动或晃动",
    "零件的对称度": "得分9: α+β<360° | 得分3: 360°≤α+β<540° | 得分1: 540°≤α+β<720°",
    "零件的重量G": "得分9: 0.1g <= G <= 2kg | 得分3: 0.01g < G < 0.1g 或 2kg < G <= 6kg | 得分1: G <= 0.01g 或 G > 6kg",
    "零件最长边长度L": "得分9: 5mm < L < 50mm | 得分3: 2mm < L < 5mm或50mm < L < 200mm | 得分1: L < 2mm或 L> 200mm",
    "零件夹持便利性": "得分9: 零件具有可夹持面且可复用现有夹持器 | 得分3: 零件具有可夹持面但需要专用夹持器 | 得分1: 零件不具有可夹持面",
    "装配动作": "得分9: 简单的直线按压或插入动作 | 得分3: 需要旋转的动作（如拧螺丝） | 得分1: 需要同时协调多个零件的复杂动作",
    "装配操作可及性": "得分9: 装配路径无阻碍且方向与主流向一致 | 得分3: 装配方向与主流向不同，需要机器人调整姿态 | 得分1: 装配空间受限，且需要使用特殊工具",
    "插入引导特征": "得分9: 零件带有倒角或圆角以辅助插入 | 得分3: 无倒角，但有其他锥形或曲面特征辅助插入 | 得分1: 无任何倒角或引导特征",
    "公差范围T": "得分9: T > 0.5 mm | 得分3: 0.1 mm ≤ T ≤ 0.5 mm | 得分1: T < 0.1 mm",
    "装配后的自保持性": "得分9: 零件装入后立即自锁或稳固固定 | 得分3: 零件能靠重力或凹槽保持位置，但未固定 | 得分1: 装入后若不扶持，零件会掉落或移位",
    "紧固方法": "得分9: 无需紧固件（如卡扣、过盈配合） | 得分3: 使用标准紧固件（如螺丝、铆钉） | 得分1: 采用永久性连接（如焊接、胶合）",
    "装配所需辅助设备": "得分9: 无需任何工具或辅助设备即可装配 | 得分3: 需要简单的通用工具辅助（如螺丝刀、扳手） | 得分1: 需要大型或专用的辅助设备（如压力机、焊接机器人）",
    "检查/调整": "得分9: 无需检查零件是否到位 (对应6轴及以上) | 得分3: 有必要检查零件是否到位或装配正确 (对应3-5轴) | 得分1: 需要调整或重新定位零件 (对应3轴以下)"
}
DFA_CRITERIA_LIST = list(DFA_SCORING_TABLE.keys())

SCORING_AGENT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:
{tools}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: a JSON dictionary as a string, with arguments for the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now have the final score for this single criterion.
Final Answer: the final JSON result from the dfa_scoring_tool
Begin!
## System Instructions:
你是一个严谨、专注的DFA评分机器人。你的唯一任务是根据用户的'Question'，为【一个指定的评分项】进行打分。
### ★★★ 核心工作原则 (必须严格遵守) ★★★
1.  **单点聚焦**: 你的每一次执行都只为了一个评分项。在思考和行动时，必须完全忽略与其他评分项相关的信息。
2.  **信息纯净**: 在调用 `dfa_scoring_tool` 时，`part_parameters` 的值**必须只包含**与当前 `criterion_name` 直接相关的信息。**绝对不要**包含其他无关评分项的数据。
### ★★★ 单项评分工作流程 ★★★
1.  **识别任务**: 从'Question'中识别出你需要评估的【评分项名称】和【已知的零件信息】。
2.  **思考所需信息**:
    *   **【特殊规则】如果评分项是 "检查/调整"**: 你【必须只】获取操作设备的【轴数】。
    *   **【特殊规则】如果评分项是 "装配动作"**: 你【必须只】获取并提供完整的【设备型号名称】。
    *   **【特殊规则】如果评分项是 "零件夹持便利性"**: 你【必须只】获取操作设备的【手部型号 (RobotHandType)】。
    *   **【特殊规则】如果评分项是 "紧固方法"或"装配动作"**: 你【必须只】看操作设备的全名分析后缀即可，无需调用工具。
    *   **对于其他评分项**: 思考评估它【只】需要什么信息，例如评估中给出了属于最少零件直接对应评估规则9分后直接输出，不需要调用工具的评分选项就直接输出评分规则。
3.  **信息收集 (如果需要)**:
    *   如果缺少【精确的技术参数】（如轴数），调用 `analytical_query_tool`。
    *   如果缺少【描述性信息】（如功能），调用 `retrieval_qa_tool`。
4.  **最终评分 (调用 dfa_scoring_tool)**:
    *   准备 Action Input 时，严格遵守【信息纯净】原则，`part_parameters` 中只放入当前评分项所需的最少信息。
Question: {input}
Thought:{agent_scratchpad}"""

llm = ChatOpenAI(model=ROUTER_LLM_MODEL_NAME, openai_api_base=ROUTER_API_BASE_URL, openai_api_key=ROUTER_API_KEY,
                 temperature=0.0, model_kwargs={"stop": ["\nObservation:"]})
scoring_tools = [dfa_scoring_tool, retrieval_qa_tool, analytical_query_tool]
scoring_prompt = PromptTemplate.from_template(SCORING_AGENT_TEMPLATE)
scoring_agent = create_react_agent(llm, scoring_tools, scoring_prompt)
scoring_agent_executor = AgentExecutor(agent=scoring_agent, tools=scoring_tools, verbose=True,
                                       handle_parsing_errors=True)

def run_dfa_evaluation(user_input: str):
    if not user_input.strip():
        yield "请输入有效的零件信息。"
        return
    output_stream = ""
    output_stream += f"**您输入:**\n> {user_input}\n\n---\n"
    yield output_stream

    log_stream = io.StringIO()
    with contextlib.redirect_stdout(log_stream):
        evaluation_results = []
        for criterion in DFA_CRITERIA_LIST:
            chunk = f"### 主管：开始评估 '{criterion}' ...\n"
            output_stream += chunk
            yield output_stream

            task_input = f"请为评分项 '{criterion}' 进行打分。已知的全部信息如下：'{user_input}'"
            try:
                result_dict = scoring_agent_executor.invoke({"input": task_input})
                scoring_json_str = result_dict['output']
                score_data = json.loads(scoring_json_str)

                agent_thought_process = log_stream.getvalue();
                log_stream.truncate(0);
                log_stream.seek(0)
                chunk = f"<details><summary>点击查看 '{criterion}' 评分过程</summary>\n\n```log\n{agent_thought_process}\n```\n\n</details>\n"
                output_stream += chunk
                yield output_stream

                if score_data.get('score') == 1:
                    chunk = f"#### 主管决策：'{criterion}' 得分低，正在调用专家工具获取建议...\n"
                    output_stream += chunk
                    yield output_stream

                    part_specific_info = score_data.get('value', '未知参数')
                    rule_context = DFA_SCORING_TABLE.get(criterion, "无特定规则说明。")
                    focused_part_info = (
                        f"一个零件在 '{criterion}' 方面的具体参数是 '{part_specific_info}'。" f"我们的评分标准是：【{rule_context}】。")
                    suggestion = suggestion_expert_tool.invoke(
                        {"part_info": focused_part_info, "low_score_item_json": scoring_json_str})
                    score_data['suggestion'] = suggestion

                    chunk = f"**专家建议**:\n{suggestion}\n\n---\n"
                    output_stream += chunk
                    yield output_stream

                evaluation_results.append(score_data)
            except Exception as e:
                chunk = f"<p style='color:red;'>评估 '{criterion}' 时出错: {e}</p>\n"
                output_stream += chunk
                yield output_stream
                evaluation_results.append(
                    {"criterion": criterion, "score": "错误", "reasoning": f"评估失败: {e}", "value": "N/A"})

    final_report = "## 最终DFA评估报告\n\n"
    final_report += f"**评估对象:** {user_input}\n\n"
    final_report += "| 评分项 | 评估参数 | 得分 | 打分理由 |\n|---|---|---|---|\n"
    total_score = 0
    suggestions = []
    for result in evaluation_results:
        score = result.get('score', 'N/A')
        final_report += f"| {result.get('criterion', 'N/A')} | {result.get('value', 'N/A')} | **{score}** | {result.get('reasoning', 'N/A')} |\n"
        if isinstance(score, (int, float)):
            total_score += score
        if 'suggestion' in result:
            suggestions.append(f"*   **针对【{result['criterion']}】(得分: {score}):**\n    {result['suggestion']}")
    final_report += f"\n**评估总分: {total_score}**\n"
    if suggestions:
        final_report += "\n---\n### **专家改进建议**\n\n" + "\n\n".join(suggestions)

    output_stream += final_report
    yield output_stream

with gr.Blocks(title="智能DFA评估系统") as demo:
    gr.Markdown("# 智能DFA评估系统-DEMO")
    gr.Markdown("输入零件的详细描述，系统将自动进行多维度DFA评估，并对低分项给出专家建议。")

    with gr.Row():
        msg = gr.Textbox(
            label="输入零件信息",
            placeholder="例如：一个螺栓它的合格率为98%（缺陷水平P=2%），重量为8kg...",
            show_label=False,
            container=False,
            scale=8,
        )
        submit_btn = gr.Button("提交评估", variant="primary", scale=1)
        clear_btn = gr.Button("清除", scale=1)

    report_display = gr.Markdown(label="评估过程与报告")

    def clear_all():
        return "", ""

    submit_btn.click(run_dfa_evaluation, inputs=[msg], outputs=[report_display])
    msg.submit(run_dfa_evaluation, inputs=[msg], outputs=[report_display])

    clear_btn.click(clear_all, None, [msg, report_display], queue=False)

demo.launch(share=True, server_name="0.0.0.0")