from models.base import ProcessingContext
from agents.query_analyzer import QueryAnalyzerAgent
from agents.movie_search import MovieSearchAgent
from agents.person_search import PersonSearchAgent
from agents.general_knowledge import GeneralKnowledgeAgent
from agents.response_formatter import ResponseFormatterAgent
from tools.movie_api import MovieAPITool
from tools.vector_db import MovieVectorDatabaseTool
from tools.general_knowledge import GeneralKnowledgeTool
from tools.llm_interface import LLMInterfaceTool
from models import ToolManager, Tool
from models import (
    Agent,
    FormattedResponse,
    QueryAnalysis,
    GeneralKnowledgeResult,
    MovieRecommendationResult,
)
from termcolor import colored
from typing import Dict, List, Optional, Any
from datetime import datetime


# ==================== ORCHESTRATOR ====================
class Orchestrator:
    """Orchestrator that manages the agentic workflow with centralized tool assignment."""

    def __init__(self):
        self.tool_manager = ToolManager()
        self._setup_tools()
        self.tool_manager.initialize_all()
        self.agent_configurations: Dict[str, List[str]] = {
            "analyzer": ["LLMInterfaceTool"],
            "movie_searcher": [
                "MovieAPITool",
                "MovieVectorDatabaseTool",
                "LLMInterfaceTool",
            ],
            "person_searcher": ["MovieAPITool", "LLMInterfaceTool"],
            "general_knowledge": ["GeneralKnowledgeTool", "LLMInterfaceTool"],
            "formatter": ["LLMInterfaceTool", "MovieAPITool"],
        }
        self.agents: Dict[str, Agent] = {
            "analyzer": QueryAnalyzerAgent(),
            "movie_searcher": MovieSearchAgent(),
            "person_searcher": PersonSearchAgent(),
            "general_knowledge": GeneralKnowledgeAgent(),
            "formatter": ResponseFormatterAgent(),
        }
        self.agent_available_tools: Dict[str, List[Tool]] = {}
        self._prepare_agent_tools()

    def _setup_tools(self):
        """Setup and register tools with the ToolManager."""
        tools = [
            MovieAPITool(),
            MovieVectorDatabaseTool(),
            GeneralKnowledgeTool(),
            LLMInterfaceTool(),
        ]
        for tool in tools:
            self.tool_manager.register_tool(tool)

    def _prepare_agent_tools(self):
        """Pre-fetch the list of available tools for each agent based on configuration."""
        for agent_name, required_tool_names in self.agent_configurations.items():
            available_tools_for_agent = []
            for tool_name in required_tool_names:
                tool_instance = self.tool_manager.get_tool(tool_name)
                if tool_instance:
                    available_tools_for_agent.append(tool_instance)
            self.agent_available_tools[agent_name] = available_tools_for_agent
        print(colored("[ORCHESTRATOR] Agent tool assignments prepared.", "blue"))

    def _select_next_agent(self, context: ProcessingContext) -> Optional[str]:
        """
        Selects the next agent to execute based on the current context state.
        This is the core of the agentic decision-making.
        """
        # Goal: Produce a final_response in formatted_response.
        # Check if the goal is already met or if we are in an unrecoverable error state.
        if context.formatted_response and context.formatted_response.final_response:
            # Goal achieved or seems to be achieved (assuming formatter success means task done)
            # unless we explicitly want to add a review/reflect step.
            print(
                colored(
                    "[ORCHESTRATOR] Goal appears achieved (formatted response exists).",
                    "green",
                )
            )
            return None  # Signal to stop the loop

        # --- Decision Logic based on Context State ---

        # 1. If no analysis, we must analyze the query first.
        if not context.analysis:
            print(
                colored(
                    "[ORCHESTRATOR] No analysis found. Selecting QueryAnalyzer.", "blue"
                )
            )
            return "analyzer"

        # 2. If analysis exists, determine the next step based on topic and current results.
        analysis = context.analysis
        topic = analysis.topic

        # Check if the expected result type for the topic already exists.
        # If it does, we might have completed the main task for that topic.
        # If it doesn't, we need to execute the corresponding agent.

        if topic == "movie":
            if context.movie_result is None:
                print(
                    colored(
                        "[ORCHESTRATOR] Analysis is 'movie' but no movie result. Selecting MovieSearcher.",
                        "blue",
                    )
                )
                return "movie_searcher"
            elif (
                context.movie_result
                and isinstance(context.movie_result, dict)
                and "error" in context.movie_result
            ):
                # Error occurred in movie search, might need to retry or handle differently.
                # For simplicity in this loop, we'll proceed to format the error.
                # A more advanced loop could decide to retry or ask for clarification.
                print(
                    colored(
                        "[ORCHESTRATOR] Movie search resulted in error. Proceeding to Formatter.",
                        "yellow",
                    )
                )
                return "formatter"
            else:
                # Movie result exists (success or structured data). Check if formatted.
                if context.formatted_response is None:
                    print(
                        colored(
                            "[ORCHESTRATOR] Movie result found. Selecting Formatter.",
                            "blue",
                        )
                    )
                    return "formatter"
                else:
                    # Result exists and is formatted. Goal likely met.
                    return None

        elif topic == "person":
            if context.person_result is None:
                print(
                    colored(
                        "[ORCHESTRATOR] Analysis is 'person' but no person result. Selecting PersonSearcher.",
                        "blue",
                    )
                )
                return "person_searcher"
            elif (
                context.person_result
                and isinstance(context.person_result, dict)
                and "error" in context.person_result
            ):
                print(
                    colored(
                        "[ORCHESTRATOR] Person search resulted in error. Proceeding to Formatter.",
                        "yellow",
                    )
                )
                return "formatter"
            else:
                if context.formatted_response is None:
                    print(
                        colored(
                            "[ORCHESTRATOR] Person result found. Selecting Formatter.",
                            "blue",
                        )
                    )
                    return "formatter"
                else:
                    return None

        elif topic == "general":
            if context.general_result is None:
                print(
                    colored(
                        "[ORCHESTRATOR] Analysis is 'general' but no general result. Selecting GeneralKnowledge.",
                        "blue",
                    )
                )
                return "general_knowledge"
            elif (
                context.general_result
                and isinstance(context.general_result, dict)
                and "error" in context.general_result
            ):
                print(
                    colored(
                        "[ORCHESTRATOR] General knowledge search resulted in error. Proceeding to Formatter.",
                        "yellow",
                    )
                )
                return "formatter"
            else:
                if context.formatted_response is None:
                    print(
                        colored(
                            "[ORCHESTRATOR] General result found. Selecting Formatter.",
                            "blue",
                        )
                    )
                    return "formatter"
                else:
                    return None

        # --- Fallback Logic ---
        # If state is somehow ambiguous or doesn't match clear paths,
        # we can use the priority order or default to formatting if any result exists.
        print(
            colored(
                "[ORCHESTRATOR] State ambiguous or unhandled. Defaulting to Formatter if any result exists, otherwise stopping.",
                "yellow",
            )
        )
        if (
            context.movie_result
            or context.person_result
            or context.general_result
            or context.formatted_response
        ):
            return "formatter"
        # If nothing is set, and we've analyzed, maybe we should format an error or stop.
        # Let's stop to prevent loops if state is truly broken.
        return None

    def execute_agent(
        self, agent_name: str, context: ProcessingContext
    ) -> Dict[str, Any]:
        """Execute a specific agent with error handling and pre-assigned available tools."""
        if agent_name in self.agents:
            agent = self.agents[agent_name]
            print(colored(f"[ORCHESTRATOR] Executing {agent.name}...", "magenta"))
            try:
                available_tools_for_this_agent = self.agent_available_tools.get(
                    agent_name, []
                )
                required_tool_names = self.agent_configurations.get(agent_name, [])
                result = agent.execute(
                    context,
                    self.tool_manager,
                    available_tools_for_this_agent,
                    required_tool_names,
                )
                print(colored(f"[ORCHESTRATOR] {agent.name} completed.", "magenta"))
                return result
            except Exception as e:
                import traceback

                print(traceback.format_exc())  # Print full traceback for debugging
                print(colored(f"[ORCHESTRATOR] Error in {agent.name}: {e}", "red"))
                # Return a standardized error result that the loop can understand
                return {
                    "error": f"Error in {agent.name}: {str(e)}",
                    "agent_name": agent_name,
                }
        else:
            print(
                colored(f"[ORCHESTRATOR] Warning: Agent {agent_name} not found!", "red")
            )
            return {"error": f"Agent {agent_name} not found", "agent_name": agent_name}

    def process_request(self, user_input: str) -> Dict[str, Any]:
        """Main method to process user request using an agentic loop."""
        print(
            colored(
                f"[ORCHESTRATOR] Starting agentic loop for request: '{user_input}'",
                "magenta",
            )
        )

        # Initialize context
        context_model = ProcessingContext(
            user_input=user_input, timestamp=datetime.now().isoformat()
        )

        # Agentic Loop Parameters
        max_iterations = 10  # Prevent infinite loops
        iteration_count = 0

        # --- MAIN AGENTIC LOOP ---
        while iteration_count < max_iterations:
            iteration_count += 1
            print(
                colored(
                    f"[ORCHESTRATOR] --- Loop Iteration {iteration_count} ---", "cyan"
                )
            )

            # 1. OBSERVE: Check the current state of the context
            # (This is implicit in _select_next_agent, but we could log it here)
            # print(f"[DEBUG] Current Context State: Analysis={context_model.analysis is not None}, MovieRes={context_model.movie_result is not None}, ...")

            # 2. DECIDE: Select the next agent to run based on the observed state
            next_agent_name = self._select_next_agent(context_model)

            # 3. CHECK TERMINATION: If no agent is selected, the goal is likely met or an error state is reached.
            if not next_agent_name:
                print(
                    colored(
                        "[ORCHESTRATOR] No next agent selected. Goal likely achieved or max iterations reached. Exiting loop.",
                        "green",
                    )
                )
                break

            print(
                colored(
                    f"[ORCHESTRATOR] Selected next agent: {next_agent_name}", "blue"
                )
            )

            # 4. ACT: Execute the selected agent
            agent_result = self.execute_agent(next_agent_name, context_model)

            # 5. UPDATE STATE: Update the context model based on the agent's result
            # This is crucial for the loop's next decision.
            try:
                if next_agent_name == "analyzer" and "analysis" in agent_result:
                    # Validate and update analysis
                    if isinstance(agent_result["analysis"], QueryAnalysis):
                        context_model.analysis = agent_result["analysis"]
                    else:
                        # If it's a dict, try to parse it
                        context_model.analysis = QueryAnalysis(
                            **agent_result["analysis"]
                        )

                elif (
                    next_agent_name == "movie_searcher"
                    and "movie_result" in agent_result
                ):
                    # Validate and update movie result
                    if isinstance(
                        agent_result["movie_result"], MovieRecommendationResult
                    ):
                        context_model.movie_result = agent_result["movie_result"]
                    elif (
                        isinstance(agent_result["movie_result"], dict)
                        and "error" in agent_result["movie_result"]
                    ):
                        # Handle error dict if returned directly
                        context_model.movie_result = agent_result[
                            "movie_result"
                        ]  # Keep as dict for now
                    else:
                        # If it's a dict, try to parse it (assuming successful result structure)
                        try:
                            context_model.movie_result = MovieRecommendationResult(
                                **agent_result["movie_result"]
                            )
                        except Exception as e:
                            print(
                                colored(
                                    f"[ORCHESTRATOR] Error parsing movie_searcher result: {e}. Keeping as dict.",
                                    "red",
                                )
                            )
                            context_model.movie_result = agent_result[
                                "movie_result"
                            ]  # Fallback to dict

                elif (
                    next_agent_name == "person_searcher"
                    and "person_result" in agent_result
                ):
                    if isinstance(agent_result["person_result"], PersonSearchResult):
                        context_model.person_result = agent_result["person_result"]
                    elif (
                        isinstance(agent_result["person_result"], dict)
                        and "error" in agent_result["person_result"]
                    ):
                        context_model.person_result = agent_result["person_result"]
                    else:
                        try:
                            context_model.person_result = PersonSearchResult(
                                **agent_result["person_result"]
                            )
                        except Exception as e:
                            print(
                                colored(
                                    f"[ORCHESTRATOR] Error parsing person_searcher result: {e}. Keeping as dict.",
                                    "red",
                                )
                            )
                            context_model.person_result = agent_result["person_result"]

                elif (
                    next_agent_name == "general_knowledge"
                    and "general_result" in agent_result
                ):
                    if isinstance(
                        agent_result["general_result"], GeneralKnowledgeResult
                    ):
                        context_model.general_result = agent_result["general_result"]
                    elif (
                        isinstance(agent_result["general_result"], dict)
                        and "error" in agent_result["general_result"]
                    ):
                        context_model.general_result = agent_result["general_result"]
                    else:
                        try:
                            context_model.general_result = GeneralKnowledgeResult(
                                **agent_result["general_result"]
                            )
                        except Exception as e:
                            print(
                                colored(
                                    f"[ORCHESTRATOR] Error parsing general_knowledge result: {e}. Keeping as dict.",
                                    "red",
                                )
                            )
                            context_model.general_result = agent_result[
                                "general_result"
                            ]

                elif next_agent_name == "formatter" and agent_result:
                    # The formatter returns a dict that can be used to create FormattedResponse
                    # This is the final step that should produce the output.
                    try:
                        context_model.formatted_response = FormattedResponse(
                            **agent_result
                        )
                    except ValidationError as e:
                        print(
                            colored(
                                f"[ORCHESTRATOR] Error creating FormattedResponse from formatter output: {e}",
                                "red",
                            )
                        )
                        # Create a fallback response indicating formatting error
                        context_model.formatted_response = FormattedResponse(
                            final_response=f"Error formatting response: {e}",
                            movie_recommendations=None,
                            general_info=None,
                            person_info=None,
                        )
                    except Exception as e:  # Catch other potential errors
                        print(
                            colored(
                                f"[ORCHESTRATOR] Unexpected error creating FormattedResponse: {e}",
                                "red",
                            )
                        )
                        context_model.formatted_response = FormattedResponse(
                            final_response=f"Unexpected error during formatting: {e}",
                            movie_recommendations=None,
                            general_info=None,
                            person_info=None,
                        )

                # Handle errors from agent execution itself (not errors returned *by* the agent)
                elif "error" in agent_result:
                    print(
                        colored(
                            f"[ORCHESTRATOR] Agent '{next_agent_name}' execution failed: {agent_result['error']}",
                            "red",
                        )
                    )
                    # Decide how to handle execution errors. For now, we'll try to format an error message.
                    # This could involve setting an error flag in context or directly trying the formatter.
                    # Let's try to trigger the formatter to handle this error state.
                    # The _select_next_agent logic should now choose formatter because an error dict might be in the results.
                    # Alternatively, we could explicitly set a flag or dummy result to trigger formatting.
                    # Let's set a generic error result for the topic if not already set, to guide selection.
                    if context_model.analysis:
                        topic = context_model.analysis.topic
                        if topic == "movie" and context_model.movie_result is None:
                            context_model.movie_result = {
                                "error": agent_result["error"],
                                "agent": next_agent_name,
                            }
                        elif topic == "person" and context_model.person_result is None:
                            context_model.person_result = {
                                "error": agent_result["error"],
                                "agent": next_agent_name,
                            }
                        elif (
                            topic == "general" and context_model.general_result is None
                        ):
                            context_model.general_result = {
                                "error": agent_result["error"],
                                "agent": next_agent_name,
                            }
                    else:
                        # Error before analysis, very early failure. Try to format.
                        context_model.formatted_response = FormattedResponse(
                            final_response=f"Critical error during processing: {agent_result['error']}",
                            movie_recommendations=None,
                            general_info=None,
                            person_info=None,
                        )
                        # This should cause the next loop iteration to exit.

            except Exception as e:
                print(
                    colored(
                        f"[ORCHESTRATOR] Error updating context after agent '{next_agent_name}' execution: {e}",
                        "red",
                    )
                )
                # This is an unexpected error in the orchestrator logic. Set a critical error response.
                context_model.formatted_response = FormattedResponse(
                    final_response=f"Internal orchestrator error: {e}",
                    movie_recommendations=None,
                    general_info=None,
                    person_info=None,
                )
                # Force exit on internal orchestrator error
                break

            # End of loop iteration
            print(
                colored(
                    f"[ORCHESTRATOR] --- End of Iteration {iteration_count} ---", "cyan"
                )
            )

        # --- POST-LOOP ---
        if iteration_count >= max_iterations:
            print(
                colored(
                    f"[ORCHESTRATOR] Warning: Maximum iterations ({max_iterations}) reached. Loop terminated.",
                    "red",
                )
            )
            if not context_model.formatted_response:
                context_model.formatted_response = FormattedResponse(
                    final_response="Processing timed out or reached maximum steps.",
                    movie_recommendations=None,
                    general_info=None,
                    person_info=None,
                )

        # Ensure a final response is always present
        if not context_model.formatted_response:
            context_model.formatted_response = FormattedResponse(
                final_response="Processing completed, but no response was generated.",
                movie_recommendations=None,
                general_info=None,
                person_info=None,
            )

        print(colored("[ORCHESTRATOR] Agentic loop task completed.", "magenta"))
        return context_model.dict()


# ==================== MAIN SYSTEM ====================
class UniversalAssistantSystem:
    """Main system that handles movie, person, and general queries."""

    def __init__(self):
        self.orchestrator = Orchestrator()

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Public interface for processing any query"""
        try:
            return self.orchestrator.process_request(user_query)
        except Exception as e:
            import traceback

            print(traceback.format_exc())  # Print full traceback for debugging
            return {
                "error": f"System error: {str(e)}",
                "final_response": "Sorry, I encountered an error processing your request. Please try again.",
            }

    # --- IMPORTANT: Updated chat method ---
    def chat(self, user_query: str) -> str:
        """Simplified interface that returns just the formatted response"""
        result = self.process_query(user_query)
        # print(colored(f"[UniversalAssistantSystem] Processed result: '{result}'", "green"))
        # Check if process_query returned an error dictionary with a top-level final_response
        top_level_response = result.get("final_response")
        if top_level_response:
            return top_level_response
        # Otherwise, try to get the nested final_response from the formatted result
        # Access the nested structure correctly
        nested_final_response = result.get("formatted_response", {}).get(
            "final_response"
        )
        return (
            nested_final_response
            if nested_final_response
            else "Sorry, I couldn't process that request."
        )

    # --- End of Update ---
