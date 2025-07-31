from models.base import QueryAnalysis, ProcessingContext
from tools.llm_interface import LLMInterfaceTool
from termcolor import colored
import json
import re
from pydantic import ValidationError

from models import Agent, ToolManager, Tool
from typing import Dict, Any, List, Optional


class QueryAnalyzerAgent(Agent):
    """Agent for analyzing user queries."""

    def __init__(self):
        system_prompt = """
You are an intelligent query classifier and information extractor.
Your task is to analyze a user query and respond with a JSON object containing the following fields:
1.  `topic`: The main topic of the query. Choose from: "movie", "person", "general".
2.  `intent`: The user's intent. Choose from:
    *   For `topic: "movie"`: "recommendation", "specific_info" (e.g., rating, plot)
    *   For `topic: "person"`: "information", "find_movies" (movies they are known for)
    *   For `topic: "general"`: "information"
3.  `specific_request`: A specific item related to the intent.
    *   For `intent: "recommendation"`: A genre, mood, or description (e.g., "sci-fi", "thriller with plot twists").
    *   For `intent: "specific_info"` (movie): The exact movie title (e.g., "Inception").
    *   For `intent: "information"` (person): The person's name (e.g., "Leonardo DiCaprio").
    *   For `intent: "find_movies"` (person): The person's name (e.g., "Tom Hanks").
    *   For `intent: "information"` (general): The main subject of the question (e.g., "capital of France").
4.  `entities`: A dictionary to hold extracted entities.
    *   `liked_movies`: A list of movie titles the user likes (if any).
    *   `disliked_movies`: A list of movie titles the user dislikes (if any).
    *   `desired_genres`: A list of genres the user is interested in (if any).
    *   `excluded_genres`: A list of genres the user wants to avoid (if any).
    *   `keywords`: A list of other relevant keywords from the query.
    *   `actors`: A list of actor names mentioned (especially for person topics).
CRITICAL INSTRUCTIONS:
*   Respond ONLY with the JSON object. No extra text.
*   If a field is not applicable or cannot be determined, use an appropriate default:
    *   Strings: ""
    *   Lists: []
    *   Objects: {}
*   Prioritize accuracy. If uncertain, prefer general/default values.
*   For movie/person names, use the most common or likely spelling based on the query.
Example Input: "I want to see a good sci-fi movie, like Inception."
Example Output: { "topic": "movie", "intent": "recommendation", "specific_request": "sci-fi", "entities": { "liked_movies": ["Inception"], "keywords": ["good"] } }
Example Input: "What is the rating of Inception?"
Example Output: { "topic": "movie", "intent": "specific_info", "specific_request": "Inception", "entities": {} }
Example Input: "Tell me about The Matrix."
Example Output: { "topic": "movie", "intent": "specific_info", "specific_request": "The Matrix", "entities": {} }
CRITICAL INSTRUCTION FOR person topic:
If the topic is "person", and the intent is "find_movies" or "information", extract the person's name into the "actors" list within "entities".
If the intent is "find_movies", the "specific_request" field should be empty.
""".strip()
        super().__init__("QueryAnalyzer", system_prompt)

    def execute(
        self,
        context: ProcessingContext,
        tool_manager: ToolManager,
        available_tools: List[Tool],
        required_tool_names: List[str],
    ) -> Dict[str, Any]:
        user_query = context.user_input
        print(colored(f"[QueryAnalyzer] Analyzing query: '{user_query}'", "cyan"))
        llm_tool: Optional[LLMInterfaceTool] = next(
            (t for t in available_tools if isinstance(t, LLMInterfaceTool)), None
        )
        if not llm_tool:
            print(
                colored("[QueryAnalyzer] Error: LLMInterfaceTool not available.", "red")
            )
            return {}
        try:
            prompt = f"Query: {user_query}\nRespond with the JSON object."
            response = llm_tool.generate(prompt, self.system_prompt)
            if "<think>" in response and "</think>" in response:
                response_thinking = (
                    response.split("<think>")[1].split("</think>")[0].strip()
                )
                response = response.split("</think>")[-1].strip()
                print(
                    colored(
                        f"[QueryAnalyzer] LLM response thinking: {response_thinking}",
                        "green",
                    )
                )

            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                json_str = response[start:end]
                parsed_data = json.loads(json_str)
                analysis_model = QueryAnalysis(**parsed_data)
                return {"analysis": analysis_model}
            else:
                raise ValueError("No valid JSON object found in LLM response")
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            print(
                colored(
                    f"[QueryAnalyzer] Error parsing/validating LLM response: {e}. Response was: {response}",
                    "red",
                )
            )
            query_lower = user_query.lower()
            movie_keywords = [
                "movie",
                "film",
                "watch",
                "cinema",
                "theater",
                "rating",
                "director",
                "plot",
            ]
            person_keywords = [
                "actor",
                "actress",
                "star",
                "cast",
                "played",
                "movies with",
                "films with",
            ]
            if any(word in query_lower for word in person_keywords):
                actor_match = re.search(
                    r"(?:who\s+is\s+|tell\s+me\s+about\s+|movies\s+with\s+)([\w\s]+)",
                    query_lower,
                )
                if actor_match:
                    actor_name = actor_match.group(1).strip()
                    fallback_analysis = QueryAnalysis(
                        topic="person",
                        intent="information",
                        specific_request=actor_name,
                        entities={"actors": [actor_name]},
                    )
                else:
                    fallback_analysis = QueryAnalysis(
                        topic="person",
                        intent="information",
                        specific_request="",
                        entities={},
                    )
            elif any(word in query_lower for word in movie_keywords) or re.search(
                r'"([^"]+)"', user_query
            ):
                fallback_analysis = QueryAnalysis(
                    topic="movie",
                    intent="specific_info",
                    specific_request=user_query,
                    entities={},
                )
            else:
                fallback_analysis = QueryAnalysis(
                    topic="general",
                    intent="information",
                    specific_request=user_query,
                    entities={},
                )
            print(
                colored(
                    f"[QueryAnalyzer] Using fallback analysis: {fallback_analysis}",
                    "yellow",
                )
            )
            return {"analysis": fallback_analysis}
        except Exception as e:
            print(colored(f"[QueryAnalyzer] Unexpected error: {e}", "red"))
            return {}
