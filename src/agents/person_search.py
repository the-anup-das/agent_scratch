from models.base import (
    PersonSearchResult,
    PersonResult,
    PersonData,
    MovieData,
    ProcessingContext,
)
from tools.movie_api import MovieAPITool
from tools.llm_interface import LLMInterfaceTool
from termcolor import colored
from pydantic import ValidationError
import re
from models import Agent


class PersonSearchAgent(Agent):
    """Agent for searching person information."""

    def __init__(self):
        self.system_prompt = "You are a person information assistant."

    def execute(
        self,
        context: ProcessingContext,
        tool_manager,
        available_tools,
        required_tool_names,
    ):
        analysis = context.analysis
        if not analysis or analysis.topic != "person":
            return {}
        intent = analysis.intent
        specific_request = analysis.specific_request
        entities = analysis.entities
        user_query = context.user_input
        print(
            colored(
                f"[PersonSearcher] Intent: {intent}, Request: '{specific_request}'",
                "cyan",
            )
        )
        api_tool = next(
            (t for t in available_tools if isinstance(t, MovieAPITool)), None
        )
        llm_tool = next(
            (t for t in available_tools if isinstance(t, LLMInterfaceTool)), None
        )
        if not api_tool:
            error_msg = "[PersonSearcher] Error: MovieAPITool (mandatory for person data) not available."
            print(colored(error_msg, "red"))
            return {"error": error_msg}
        person_name = ""
        if intent in ["information", "find_movies"]:
            if entities.get("actors"):
                person_name = entities["actors"][0]
            elif specific_request:
                person_name = specific_request
            else:
                name_match = re.search(
                    r"(?:who\s+is\s+|tell\s+me\s+about\s+|movies\s+with\s+)([\w\s]+)",
                    user_query.lower(),
                )
                if name_match:
                    person_name = name_match.group(1).strip()
                else:
                    person_name = user_query
        if not person_name:
            print(
                colored(
                    "[PersonSearcher] Could not determine person name from query.",
                    "red",
                )
            )
            return {"error": "Could not determine person name."}
        print(
            colored(f"[PersonSearcher] Searching for person: '{person_name}'", "cyan")
        )
        tmdb_person_results = api_tool.search_person(person_name)
        if not tmdb_person_results:
            return {"error": f"Person '{person_name}' not found."}
        person_id = tmdb_person_results[0]["id"]
        person_details = api_tool.get_person_details(person_id)
        if not person_details:
            return {"error": f"Could not fetch details for person ID {person_id}."}
        person_data_dict = {
            "id": person_details.get("id", person_id),
            "name": person_details.get("name", person_name),
            "biography": person_details.get("biography", "Biography not available."),
            "birth_date": person_details.get("birthday", ""),
            "death_date": person_details.get("deathday"),
            "place_of_birth": person_details.get("place_of_birth", ""),
            "profile_path": person_details.get("profile_path", ""),
            "known_for_department": person_details.get("known_for_department", ""),
            "popularity": person_details.get("popularity", 0.0),
            "imdb_id": person_details.get("imdb_id"),
            "source": "TMDb",
        }
        try:
            person_data = PersonData(**person_data_dict)
        except ValidationError as e:
            print(colored(f"[PersonSearcher] Error validating person  {e}", "red"))
            return {"error": "Failed to process person information."}
        known_for_movies = []
        if intent == "find_movies":
            print(
                colored(
                    f"[PersonSearcher] Finding movies for {person_data.name}...", "cyan"
                )
            )
            movie_credits = api_tool.get_person_movie_credits(person_id)
            known_for_list = sorted(
                movie_credits, key=lambda x: x.get("popularity", 0), reverse=True
            )[:10]
            for movie in known_for_list:
                try:
                    movie_data_dict = {
                        "id": movie.get("id", hash(movie.get("title", "Unknown"))),
                        "title": movie.get("title", "Unknown Title"),
                        "year": (
                            movie.get("release_date", "")[:4]
                            if movie.get("release_date")
                            else "Unknown Year"
                        ),
                        "rating": "N/A",
                        "plot": "No plot summary available.",
                        "poster_path": movie.get("poster_path", ""),
                        "genres": [],
                        "cast": [],
                        "director": "",
                        "runtime": "",
                        "tmdb_id": movie.get("id"),
                        "imdb_id": movie.get("imdb_id", ""),
                        "source": "TMDb Credit",
                    }
                    movie_model = MovieData(**movie_data_dict)
                    known_for_movies.append(movie_model)
                except ValidationError as e:
                    print(
                        colored(
                            f"[PersonSearcher] Error validating known-for movie {movie.get('title', 'Unknown')}: {e}",
                            "red",
                        )
                    )
                except Exception as e:
                    print(
                        colored(
                            f"[PersonSearcher] Unexpected error validating known-for movie: {e}",
                            "red",
                        )
                    )
            known_for_movies = known_for_movies[:5]
        person_result_data = PersonResult(
            person=person_data, known_for_movies=known_for_movies
        )
        result_model = PersonSearchResult(topic="person", result=person_result_data)
        return {"person_result": result_model}
