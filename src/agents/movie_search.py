from models import MovieRecommendationResult, MovieData, ProcessingContext
from tools.movie_api import MovieAPITool
from tools.vector_db import MovieVectorDatabaseTool
from tools.llm_interface import LLMInterfaceTool
from termcolor import colored
from pydantic import ValidationError
from typing import Dict, Any
from typing import List, Optional
from models import Tool, ToolManager

from models import Agent


class MovieSearchAgent(Agent):
    """Agent for searching and recommending movies."""

    def __init__(self):
        system_prompt = "You are a movie recommendation assistant."
        super().__init__("MovieSearcher", system_prompt)

    def _handle_recommendation(
        self, user_query: str, entities: Dict[str, Any], available_tools: List[Tool]
    ) -> Dict[str, Any]:
        """
        Handle recommendation requests using hybrid search and API enrichment.
        """
        print(
            colored(
                f"[MovieSearcher] Handling recommendation for: '{user_query}' with entities: {entities}",
                "cyan",
            )
        )
        # --- Step 1: Get Available Tools ---
        vector_db_tool: Optional[MovieVectorDatabaseTool] = next(
            (t for t in available_tools if isinstance(t, MovieVectorDatabaseTool)), None
        )
        api_tool: Optional[MovieAPITool] = next(
            (t for t in available_tools if isinstance(t, MovieAPITool)), None
        )
        if not vector_db_tool or not api_tool:
            error_msg = "[MovieSearcher] Required tools (VectorDB or API) not available for recommendation."
            print(colored(error_msg, "red"))
            return {"error": error_msg}  # Return error in expected format
        # --- Step 2: Extract Context ---
        # Note: Key names might differ slightly in your entities dict based on the analyzer
        liked_movies = entities.get("liked_movies", []) or entities.get(
            "liked_titles", []
        )
        disliked_movies = entities.get("disliked_movies", []) or entities.get(
            "disliked_titles", []
        )
        desired_genres = entities.get("genres", []) or entities.get(
            "desired_genres", []
        )
        specific_request = entities.get(
            "specific_request", user_query
        )  # Fallback to user query
        # --- Step 3: Perform Hybrid Search ---
        print(colored("[MovieSearcher] Performing hybrid search...", "cyan"))
        # Use the specific request derived from entities for better relevance
        hybrid_results = vector_db_tool.hybrid_search(
            specific_request, k=20, alpha=0.7
        )  # Favor semantic slightly
        # --- Step 4: Filter Candidates ---
        print(colored("[MovieSearcher] Filtering candidates...", "cyan"))
        disliked_titles_lower = {title.lower() for title in disliked_movies}
        # Filter out disliked movies
        filtered_candidates = [
            movie
            for movie in hybrid_results
            if movie.get("title", "").lower() not in disliked_titles_lower
        ][
            :30
        ]  # Limit for API calls
        print(
            colored(
                f"[MovieSearcher] Found {len(filtered_candidates)} candidates after filtering.",
                "cyan",
            )
        )
        # --- Step 5: Contextual Reranking (Optional but Improved) ---
        # Rerank based on liked movie context if provided, otherwise use original query
        candidates_to_enrich = filtered_candidates
        if liked_movies:
            # Combine liked movie plots for context-based reranking
            liked_plots = []
            for title in liked_movies:
                # Simple lookup in vector db metadata for plot (could be improved)
                # Access metadata through the tool
                if hasattr(vector_db_tool, "db") and hasattr(
                    vector_db_tool.db, "metadata"
                ):
                    matched_movies = [
                        m
                        for m in vector_db_tool.db.metadata
                        if m.get("title", "").lower() == title.lower()
                    ]
                    if matched_movies:
                        liked_plots.append(matched_movies[0].get("plot", ""))
            if liked_plots:
                liked_context = " ".join(liked_plots)
                print(
                    colored(
                        f"[MovieSearcher] Reranking based on liked movie context: '{liked_context[:50]}...'",
                        "cyan",
                    )
                )
                candidates_to_enrich = vector_db_tool.rerank_results(
                    liked_context, filtered_candidates
                )
            else:
                print(
                    colored(
                        "[MovieSearcher] No plots found for liked movies, using hybrid results.",
                        "yellow",
                    )
                )
        else:
            # No liked movies context, keep hybrid results
            print(
                colored(
                    "[MovieSearcher] No liked movie context provided, using hybrid search results.",
                    "cyan",
                )
            )
        # --- Step 6: Enrich with API Data ---
        print(
            colored(
                "[MovieSearcher] Enriching candidates with API data (TMDB/OMDB)...",
                "cyan",
            )
        )
        enriched_candidates = []
        for movie in candidates_to_enrich:  # Process all candidates for better sorting
            movie_id = movie.get("id")
            title = movie.get("title", "Unknown Title")
            tmdb_id = movie.get(
                "tmdb_id"
            )  # Check if TMDB ID is already in vector DB metadata
            # --- Attempt TMDB Enrichment ---
            tmdb_success = False
            try:
                # Prefer using tmdb_id if available in metadata, otherwise fallback to search by title
                # Adjust based on your MovieAPITool's actual methods
                tmdb_details = None
                if tmdb_id:
                    tmdb_details = api_tool.get_movie_details(
                        tmdb_id
                    )  # Assuming get_movie_details can take TMDB ID
                else:
                    # Search TMDB by title and take the first result's details
                    # This might require a separate search method or logic in get_movie_details
                    # For now, assume get_movie_details handles title search if ID is not numeric/valid
                    tmdb_details = api_tool.get_movie_details(title)
                if (
                    tmdb_details
                    and isinstance(tmdb_details, dict)
                    and "error" not in tmdb_details
                ):
                    # Update movie dict with TMDB data
                    # Map TMDB fields to your MovieData model fields
                    movie.update(
                        {
                            "year": (
                                tmdb_details.get("release_date", "")[:4]
                                if tmdb_details.get("release_date")
                                else movie.get("year", "Unknown")
                            ),
                            "rating": str(
                                tmdb_details.get(
                                    "vote_average", movie.get("rating", "N/A")
                                )
                            ),  # Ensure string
                            "plot": tmdb_details.get("overview", movie.get("plot", "")),
                            "poster_path": tmdb_details.get(
                                "poster_path", movie.get("poster_path", "")
                            ),
                            "director": ", ".join(
                                [
                                    person.get("name", "")
                                    for person in tmdb_details.get("credits", {}).get(
                                        "crew", []
                                    )
                                    if person.get("job") == "Director"
                                ][:1]
                            ),  # Simplified director extraction
                            "runtime": str(
                                tmdb_details.get("runtime", movie.get("runtime", ""))
                            ),  # Ensure string
                            "tmdb_id": tmdb_details.get("id", movie.get("tmdb_id")),
                            "imdb_id": tmdb_details.get(
                                "imdb_id", movie.get("imdb_id", "")
                            ),
                            "source": "TMDB",
                        }
                    )
                    enriched_candidates.append(movie)
                    tmdb_success = True
                else:
                    print(
                        colored(
                            f"[MovieSearcher] TMDB enrichment returned no data or error for '{title}' (ID: {movie_id}).",
                            "yellow",
                        )
                    )
            except Exception as e:
                print(
                    colored(
                        f"[MovieSearcher] Error enriching movie '{title}' (ID: {movie_id}) via TMDB: {e}.",
                        "yellow",
                    )
                )
            # --- Fallback to OMDB if TMDB failed ---
            if not tmdb_success:
                try:
                    omdb_data = api_tool.get_movie_rating(
                        title
                    )  # Fixed: Use public method instead of internal one
                    if (
                        omdb_data
                        and isinstance(omdb_data, dict)
                        and "error" not in omdb_data
                    ):
                        movie.update(
                            {
                                "year": omdb_data.get(
                                    "Year", movie.get("year", "Unknown")
                                ),
                                "rating": omdb_data.get(
                                    "imdbRating", movie.get("rating", "N/A")
                                ),
                                "plot": omdb_data.get("Plot", movie.get("plot", "")),
                                "poster_path": omdb_data.get(
                                    "Poster", movie.get("poster_path", "")
                                ),
                                "director": omdb_data.get(
                                    "Director", movie.get("director", "")
                                ),
                                "runtime": omdb_data.get(
                                    "Runtime", movie.get("runtime", "")
                                ),
                                "imdb_id": omdb_data.get(
                                    "imdbID", movie.get("imdb_id", "")
                                ),
                                # OMDB doesn't directly provide TMDB ID
                                "source": "OMDb",
                            }
                        )
                        enriched_candidates.append(movie)
                    else:
                        # Both failed, keep vector db data
                        print(
                            colored(
                                f"[MovieSearcher] OMDB enrichment also failed for '{title}'. Using vector DB data.",
                                "red",
                            )
                        )
                        # Ensure essential fields are present even if API fails
                        movie.setdefault("year", "Unknown")
                        movie.setdefault("rating", "N/A")  # Will sort lower
                        movie.setdefault(
                            "plot", movie.get("plot", "No plot available.")
                        )
                        movie.setdefault("poster_path", "")
                        movie.setdefault("director", "")
                        movie.setdefault("runtime", "")
                        movie.setdefault("tmdb_id", movie.get("tmdb_id"))
                        movie.setdefault("imdb_id", movie.get("imdb_id", ""))
                        movie["source"] = "VectorDB"
                        enriched_candidates.append(movie)
                except Exception as e:
                    print(
                        colored(
                            f"[MovieSearcher] Error enriching movie '{title}' (ID: {movie_id}) via OMDB: {e}. Using vector DB data.",
                            "red",
                        )
                    )
                    # Ensure essential fields are present even if API fails
                    movie.setdefault("year", "Unknown")
                    movie.setdefault("rating", "N/A")  # Will sort lower
                    movie.setdefault("plot", movie.get("plot", "No plot available."))
                    movie.setdefault("poster_path", "")
                    movie.setdefault("director", "")
                    movie.setdefault("runtime", "")
                    movie.setdefault("tmdb_id", movie.get("tmdb_id"))
                    movie.setdefault("imdb_id", movie.get("imdb_id", ""))
                    movie["source"] = "VectorDB"
                    enriched_candidates.append(movie)
            else:
                # TMDB succeeded, just add to list
                enriched_candidates.append(movie)  # Already updated inside TMDB block
        print(
            colored(
                f"[MovieSearcher] Enriched {len(enriched_candidates)} candidates with API data.",
                "cyan",
            )
        )
        # --- Step 7: Sort Final Recommendations ---
        print(
            colored(
                "[MovieSearcher] Sorting final recommendations by rating (and hybrid relevance)...",
                "cyan",
            )
        )

        def sort_key(movie):
            # Attempt to convert rating to float for sorting, default to 0 if N/A or invalid
            try:
                # Handle ratings like "N/A", "8.5/10", etc. Simplified: assume numeric or "N/A"
                raw_rating = movie.get("rating", "0")
                if isinstance(raw_rating, str) and "/" in raw_rating:
                    raw_rating = raw_rating.split("/")[0]
                rating = (
                    float(raw_rating) if raw_rating not in ["N/A", "", None] else 0.0
                )
            except (ValueError, TypeError):
                rating = 0.0
            # Use hybrid score if available (from initial hybrid search), otherwise similarity
            relevance_score = movie.get(
                "hybrid_score", movie.get("similarity_score", 0)
            )
            # Primary sort: Rating (descending), Secondary sort: Relevance (descending)
            return (-rating, -relevance_score)

        sorted_candidates = sorted(enriched_candidates, key=sort_key)
        print(
            colored(
                f"[MovieSearcher] Sorted candidates by rating and relevance. Total: {len(sorted_candidates)}.",
                "cyan",
            )
        )
        # --- Step 8: Select Top Recommendations ---
        final_recommendations = sorted_candidates[:10]
        # --- Step 9: Convert to Pydantic Models ---
        final_recommendations_models = []
        for movie_dict in final_recommendations:
            try:
                # Ensure defaults for Pydantic model if not already set during enrichment
                # (Most should be set, but this is a safety net)
                movie_dict.setdefault("year", "Unknown")
                movie_dict.setdefault("rating", "N/A")
                movie_dict.setdefault("plot", "")
                movie_dict.setdefault("poster_path", "")
                movie_dict.setdefault("director", "")
                movie_dict.setdefault("runtime", "")
                movie_dict.setdefault(
                    "tmdb_id", movie_dict.get("tmdb_id")
                )  # Could be None
                movie_dict.setdefault("imdb_id", movie_dict.get("imdb_id", ""))
                movie_dict.setdefault("source", "Unknown")
                # Ensure 'id' is present and correctly typed for MovieData
                if "id" not in movie_dict:
                    movie_dict["id"] = hash(
                        movie_dict.get("title", "") + movie_dict.get("year", "")
                    )  # Or generate a unique ID
                movie_model = MovieData(**movie_dict)
                final_recommendations_models.append(movie_model)
            except ValidationError as e:
                print(
                    colored(
                        f"[MovieSearcher] Error validating recommendation {movie_dict.get('title', 'Unknown')}: {e}",
                        "red",
                    )
                )
            except Exception as e:
                print(
                    colored(
                        f"[MovieSearcher] Unexpected error validating recommendation: {e}",
                        "red",
                    )
                )
        try:
            result_model = MovieRecommendationResult(
                topic="movie", result=final_recommendations_models
            )
            return {
                "topic": "movie",
                "result": result_model,
                "search_type": "recommendation",
            }
        except Exception as e:
            error_msg = (
                f"[MovieSearcher] Error creating MovieRecommendationResult model: {e}"
            )
            print(colored(error_msg, "red"))
            return {
                "topic": "movie",
                "result": {"error": error_msg},
                "search_type": "recommendation",
            }  # Wrap error in dict if needed

    def execute(
        self,
        context: ProcessingContext,
        tool_manager: ToolManager,
        available_tools: List[Tool],
        required_tool_names: List[str],
    ) -> Dict[str, Any]:
        analysis = context.analysis
        if not analysis or analysis.topic != "movie":
            return {}
        intent = analysis.intent
        specific_request = analysis.specific_request
        entities = analysis.entities
        user_query = context.user_input
        print(
            colored(
                f"[MovieSearcher] Intent: {intent}, Request: '{specific_request}'",
                "cyan",
            )
        )
        api_tool: Optional[MovieAPITool] = next(
            (t for t in available_tools if isinstance(t, MovieAPITool)), None
        )
        vector_db_tool: Optional[MovieVectorDatabaseTool] = next(
            (t for t in available_tools if isinstance(t, MovieVectorDatabaseTool)), None
        )
        llm_tool: Optional[LLMInterfaceTool] = next(
            (t for t in available_tools if isinstance(t, LLMInterfaceTool)), None
        )
        mandatory_tools_available = api_tool is not None and vector_db_tool is not None
        if not mandatory_tools_available:
            missing_names = []
            if not api_tool:
                missing_names.append("MovieAPITool")
            if not vector_db_tool:
                missing_names.append("MovieVectorDatabaseTool")
            error_msg = f"[MovieSearcher] Error: Mandatory tools not available: {', '.join(missing_names)}."
            print(colored(error_msg, "red"))
            return {"error": error_msg}
        if intent == "specific_info" and specific_request:
            print(
                colored(
                    f"[MovieSearcher] Fetching info for specific movie: '{specific_request}'",
                    "cyan",
                )
            )
            movie_data = api_tool.get_movie_rating(specific_request)
            if movie_data and "error" not in movie_data:
                try:
                    movie_model = MovieData(**movie_data)
                    result_model = MovieRecommendationResult(
                        topic="movie", result=[movie_model]
                    )
                    return {"movie_result": result_model}
                except ValidationError as e:
                    print(
                        colored(
                            f"[MovieSearcher] Error validating movie data: {e}", "red"
                        )
                    )
                    return {"error": "Failed to process movie information."}
            else:
                return {
                    "error": f"Movie '{specific_request}' not found or data unavailable."
                }
        elif intent == "recommendation":
            print(
                colored(
                    f"[MovieSearcher] Generating recommendations for: '{specific_request}'",
                    "cyan",
                )
            )
            print(colored("[MovieSearcher] Performing hybrid search...", "cyan"))
            hybrid_results = vector_db_tool.hybrid_search(specific_request, k=20)
            print(
                colored("[MovieSearcher] Filtering and enriching candidates...", "cyan")
            )
            liked_titles = [title.lower() for title in entities.get("liked_movies", [])]
            disliked_titles = [
                title.lower() for title in entities.get("disliked_movies", [])
            ]
            filtered_candidates = [
                movie
                for movie in hybrid_results
                if movie.get("title", "").lower() not in disliked_titles
            ][:30]
            print(colored("[MovieSearcher] Reranking candidates...", "cyan"))
            reranked_results = vector_db_tool.rerank_results(
                user_query, filtered_candidates
            )
            print(
                colored("[MovieSearcher] Enriching candidates with API data...", "cyan")
            )
            enriched_candidates = reranked_results
            print(
                colored(
                    "[MovieSearcher] Sorting final recommendations by rating (and relevance)...",
                    "cyan",
                )
            )

            def sort_key(movie):
                try:
                    rating = float(movie.get("rating", 0))
                except (ValueError, TypeError):
                    rating = 0.0
                relevance_score = movie.get(
                    "rerank_score",
                    movie.get("hybrid_score", movie.get("similarity_score", 0)),
                )
                return (-rating, -relevance_score)

            sorted_candidates = sorted(enriched_candidates, key=sort_key)
            print(
                colored(
                    f"[MovieSearcher] Sorted candidates by rating and relevance. Total: {len(sorted_candidates)}",
                    "cyan",
                )
            )
            final_recommendations_dicts = sorted_candidates[:10]
            final_recommendations_models = []
            for movie_dict in final_recommendations_dicts:
                try:
                    movie_dict.setdefault(
                        "id", hash(movie_dict.get("title", "Unknown"))
                    )
                    movie_dict.setdefault("year", "Unknown Year")
                    movie_dict.setdefault("rating", "N/A")
                    movie_dict.setdefault("plot", "No plot summary available.")
                    movie_dict.setdefault("poster_path", "")
                    movie_dict.setdefault("genres", [])
                    movie_dict.setdefault("cast", [])
                    movie_dict.setdefault("director", "")
                    movie_dict.setdefault("runtime", "")
                    movie_dict.setdefault("tmdb_id", None)
                    movie_dict.setdefault("imdb_id", None)
                    movie_dict.setdefault("source", "Dataset/Enriched")
                    movie_model = MovieData(**movie_dict)
                    final_recommendations_models.append(movie_model)
                except ValidationError as e:
                    print(
                        colored(
                            f"[MovieSearcher] Error validating recommendation {movie_dict.get('title', 'Unknown')}: {e}",
                            "red",
                        )
                    )
                except Exception as e:
                    print(
                        colored(
                            f"[MovieSearcher] Unexpected error validating recommendation: {e}",
                            "red",
                        )
                    )
            result_model = MovieRecommendationResult(
                topic="movie", result=final_recommendations_models
            )
            return {"movie_result": result_model}
        else:
            print(
                colored(
                    f"[MovieSearcher] Unsupported intent '{intent}' for topic 'movie'.",
                    "yellow",
                )
            )
            return {}
