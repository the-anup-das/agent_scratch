from models.base import FormattedResponse, MovieRecommendationResult, PersonSearchResult, ProcessingContext
from tools.llm_interface import LLMInterfaceTool
from tools.movie_api import MovieAPITool
from termcolor import colored
from pydantic import ValidationError

from models import Agent, ToolManager, Tool, ProcessingContext
from typing import Dict, Any, List, Optional

class ResponseFormatterAgent(Agent):
    """Agent for formatting the final response."""
    def __init__(self):
        system_prompt = """
        You are a helpful assistant that formats responses appropriately.
        For movie information, provide clear details.
        For general information, be informative and concise.
        """
        super().__init__("Formatter", system_prompt)
    def execute(self, context: ProcessingContext, tool_manager: ToolManager, available_tools: List[Tool], required_tool_names: List[str]) -> Dict[str, Any]:
        print(colored("[Formatter] Formatting final response...", "cyan"))
        movie_result_model = context.movie_result
        general_result_model = context.general_result
        person_result_model = context.person_result
        user_query = context.user_input
        formatted_response_str = ""
        movie_recommendations_data = None
        general_info_str = None
        person_info_data = None
        llm_tool: Optional[LLMInterfaceTool] = next((t for t in available_tools if isinstance(t, LLMInterfaceTool)), None)
        api_tool: Optional[MovieAPITool] = next((t for t in available_tools if isinstance(t, MovieAPITool)), None)
        if movie_result_model and isinstance(movie_result_model, MovieRecommendationResult) and movie_result_model.result:
            movies = movie_result_model.result
            if len(movies) == 1 and movies[0].source != "Dataset/Enriched":
                movie = movies[0]
                title = movie.title
                year = movie.year
                rating = movie.rating
                plot = movie.plot
                source = movie.source
                formatted_response_str = f"Here's information about '{title}':\n"
                formatted_response_str += f"- **Title:** {title}\n"
                formatted_response_str += f"- **Year:** {year}\n"
                formatted_response_str += f"- **Rating:** ⭐ {rating} (from {source})\n"
                formatted_response_str += f"- **Plot:** {plot}\n"
                # For specific info, we don't need the full list in movie_recommendations
            else:
                formatted_response_str = ""
                for i, movie in enumerate(movies, 1):
                    title = movie.title
                    year = movie.year
                    rating = movie.rating                 
                    plot = movie.plot
                    formatted_response_str += f"{colored(str(i) + '.', 'yellow')} {title} ({year}) - ⭐ {rating}/10\n"
                    formatted_response_str += f"  {plot[:200]}\n"
                formatted_query = f"""
                From the user's query, extract top 5 movie recommendations based on their preferences.
                User Query:
                {user_query}
                Extracted Movie details:
                {formatted_response_str}
                Format the response as a JSON array with exactly 5 movie objects. Each object should have:
                - title: movie title
                - year: release year
                - rating: movie rating
                - plot: brief plot summary
                Return ONLY valid JSON, no other text.
                Ensure the output is valid JSON as it will be parsed using `json.loads()` in Python
                It should be in the schema:
                [
                {{"title": "Movie Title", "year": "2023", "rating": "8.5", "plot": "Brief plot summary"}},
                ...
                ]
                """
                llm_response = llm_tool.generate(formatted_query, self.system_prompt)
                llm_response = llm_response.strip()
                import json_repair

                cleaned_final_result = json_repair.loads(llm_response)

                formatted_response_str = "Here are some movie recommendations:\n"
                for i, movie in enumerate(cleaned_final_result, 1):
                    if not isinstance(movie, dict):
                        print(colored(f"[Formatter] Invalid movie data: {movie}. Expected dict.", "red"))
                        continue
                    title = movie.get("title")
                    year = movie.get("year")
                    rating = movie.get("rating","N/A")
                    if rating is None or rating == "N/A" or rating == "None":
                        api_rating_response = api_tool.get_movie_rating(title) if api_tool else "N/A"
                        if api_rating_response:
                            rating= api_rating_response.get("rating", "N/A")
                    plot = movie.get("plot")
                    formatted_response_str += f"{colored(str(i) + '.', 'yellow')} {title} ({year}) - ⭐ {rating}/10\n"
                    formatted_response_str += f"  {plot}...\n"

                # For recommendations, pass the list
                movie_recommendations_data = movies # This is List[MovieData]
        elif person_result_model and isinstance(person_result_model, PersonSearchResult) and person_result_model.result:
             person_data = person_result_model.result.person
             known_for_movies = person_result_model.result.known_for_movies
             formatted_response_str = f"Information about {person_data.name}:\n"
             formatted_response_str += f"- **Biography:** {person_data.biography[:500]}...\n"
             if person_data.birth_date:
                  formatted_response_str += f"- **Born:** {person_data.birth_date}\n"
             if person_data.death_date:
                  formatted_response_str += f"- **Died:** {person_data.death_date}\n"
             if person_data.place_of_birth:
                  formatted_response_str += f"- **Place of Birth:** {person_data.place_of_birth}\n"
             formatted_response_str += f"- **Known For:** {person_data.known_for_department}\n"
             if known_for_movies:
                  formatted_response_str += "\n**Known for Movies:**\n"
                  for i, movie in enumerate(known_for_movies[:5], 1):
                       formatted_response_str += f"  {i}. {movie.title} ({movie.year})\n"
             # Pass the structured data
             person_info_data = person_result_model.result # This is PersonResult
        elif general_result_model and general_result_model.topic == "general" and general_result_model.result:
            formatted_response_str = f"{colored('Here\'s what I found:', 'green', attrs=['bold'])}\n{general_result_model.result}"
            general_info_str = general_result_model.result
        if not formatted_response_str:
            last_error = context.movie_result or context.person_result or context.general_result
            if isinstance(last_error, dict) and "error" in last_error:
                formatted_response_str = f"{colored('❌ Error:', 'red')} {last_error['error']}"
            else:
                formatted_response_str = f"{colored('🤔 I couldn\'t find information on that.', 'yellow')} Please try rephrasing your query."
        # --- FIX: Return the FormattedResponse Pydantic model instance directly ---
        # This ensures the Orchestrator can assign it correctly to context.formatted_response
        try:
            result_model = FormattedResponse(
                final_response=formatted_response_str,
                movie_recommendations=movie_recommendations_data, # List[MovieData] or None
                general_info=general_info_str, # str or None
                person_info=person_info_data # PersonResult or None
            )
            # Return the model's dict representation as the agent contract expects
            return result_model.dict()
        except ValidationError as e:
             print(colored(f"[Formatter] Error creating FormattedResponse model: {e}", "red"))
             # Fallback in case of formatter model error
             fallback_response = FormattedResponse(
                 final_response="Sorry, there was an error formatting the response.",
                 movie_recommendations=None,
                 general_info=None,
                 person_info=None
             )
             return fallback_response.dict()
