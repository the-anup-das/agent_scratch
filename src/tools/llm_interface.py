from typing import Optional
import os
from termcolor import colored
from models import Tool


class LLMInterface:
    """Interface for interacting with OpenAI API or local LLM."""

    def __init__(self, model: Optional[str] = "local-model"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None
        if self.model.startswith("gpt"):
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
                print(colored("[LLMInterface] Using OpenAI API.", "green"))
            except ImportError:
                print(
                    colored(
                        "[LLMInterface] OpenAI library not found. Please install it to use OpenAI API.",
                        "red",
                    )
                )
        else:
            print(
                colored(
                    "[LLMInterface] OPENAI_API_KEY not found. Attempting to connect to local LLM server...",
                    "yellow",
                )
            )
            try:
                from openai import OpenAI

                self.client = OpenAI(
                    base_url="http://localhost:1234/v1", api_key="not-needed"
                )
                self.model = "local-model"
                print(
                    colored(
                        "[LLMInterface] Connected to local LLM server at http://localhost:1234/v1",
                        "green",
                    )
                )
            except ImportError:
                print(
                    colored(
                        "[LLMInterface] OpenAI library not found. LLM features will be disabled.",
                        "red",
                    )
                )

    def generate(
        self, prompt: str, system_prompt: str = "", model: Optional[str] = None
    ) -> str:
        """Generate response using OpenAI API"""
        if not self.client:
            return self._get_fallback_response(prompt)
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            temp = (
                0.0
                if "classify" in system_prompt.lower()
                or "extract" in system_prompt.lower()
                else 0.7
            )
            response = self.client.chat.completions.create(
                model=model or self.model,
                messages=messages,
                temperature=temp,
                max_tokens=800,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(colored(f"LLM call failed: {e}", "red"))
            return self._get_fallback_response(prompt)

    def _get_fallback_response(self, prompt: str) -> str:
        if "rating" in prompt.lower() and (
            "inception" in prompt.lower() or "matrix" in prompt.lower()
        ):
            return (
                f"The movie '{prompt.split()[-1]}' is highly rated. (Fallback response)"
            )
        return f"[Fallback Response for: {prompt[:50]}...] (LLM not configured)"


class LLMInterfaceTool(Tool):
    """Tool wrapper for LLMInterface."""

    def __init__(self):
        super().__init__("LLMInterfaceTool")
        self.llm: Optional[LLMInterface] = None

    def initialize(self) -> bool:
        try:
            self.llm = LLMInterface()
            return self.is_available()
        except Exception as e:
            print(colored(f"[{self.name}] Initialization error: {e}", "red"))
            return False

    def is_available(self) -> bool:
        return self.llm is not None

    def generate(
        self, prompt: str, system_prompt: str = "", model: Optional[str] = None
    ) -> str:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.llm.generate(prompt, system_prompt, model)
