from models import Tool
from typing import Optional
from termcolor import colored


class GeneralKnowledgeToolImpl:
    """Tool for handling general questions"""

    def __init__(self):
        pass

    def search_general_knowledge(self, query: str) -> str:
        """Provide general knowledge answers"""
        formatted_query = f"""Answer the following question using your general knowledge.
Provide a clear, accurate, and concise response:
Question: {query}
Answer:""".strip()
        system_prompt = "Be helpful and informative. If you don't know the answer, suggest checking trusted online sources."
        return formatted_query, system_prompt


class GeneralKnowledgeTool(Tool):
    """Tool wrapper for GeneralKnowledgeToolImpl."""

    def __init__(self):
        super().__init__("GeneralKnowledgeTool")
        self.tool: Optional[GeneralKnowledgeToolImpl] = None

    def initialize(self) -> bool:
        try:
            self.tool = GeneralKnowledgeToolImpl()
            return True
        except Exception as e:
            print(colored(f"[{self.name}] Initialization error: {e}", "red"))
            return False

    def is_available(self) -> bool:
        return self.tool is not None

    def search_general_knowledge(self, query: str) -> str:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.tool.search_general_knowledge(query)
