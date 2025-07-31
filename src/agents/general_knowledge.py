from tools.general_knowledge import GeneralKnowledgeTool
from tools.llm_interface import LLMInterfaceTool
from termcolor import colored


from models import Agent, ToolManager, Tool, GeneralKnowledgeResult, ProcessingContext
from typing import Dict, Any, List, Optional


class GeneralKnowledgeAgent(Agent):
    """Agent for handling general knowledge questions"""

    def __init__(self):
        system_prompt = """
You are a general knowledge assistant. Provide accurate and helpful information.
""".strip()
        super().__init__("GeneralKnowledge", system_prompt)

    def execute(
        self,
        context: ProcessingContext,
        tool_manager: ToolManager,
        available_tools: List[Tool],
        required_tool_names: List[str],
    ) -> Dict[str, Any]:
        analysis = context.analysis
        if not analysis or analysis.topic != "general":
            return {}
        user_query = context.user_input
        print(colored(f"[GeneralKnowledge] Searching for: {user_query}", "cyan"))
        knowledge_tool: Optional[GeneralKnowledgeTool] = next(
            (t for t in available_tools if isinstance(t, GeneralKnowledgeTool)), None
        )
        llm_tool: Optional[LLMInterfaceTool] = next(
            (t for t in available_tools if isinstance(t, LLMInterfaceTool)), None
        )
        if not knowledge_tool or not llm_tool:
            missing = []
            if not knowledge_tool:
                missing.append("GeneralKnowledgeTool")
            if not llm_tool:
                missing.append("LLMInterfaceTool")
            print(
                colored(
                    f"[GeneralKnowledge] Error: Required tools not available: {', '.join(missing)}.",
                    "red",
                )
            )

            return {
                "error": f"GeneralKnowledgeAgent missing tools: {', '.join(missing)}"
            }
        try:
            query_prompt, sys_prompt = knowledge_tool.search_general_knowledge(
                user_query
            )
            answer = llm_tool.generate(query_prompt, sys_prompt)
            result_model = GeneralKnowledgeResult(topic="general", result=answer)
            return {"general_result": result_model}
        except Exception as e:
            print(colored(f"[GeneralKnowledge] Error processing query: {e}", "red"))
            re
