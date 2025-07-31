import os
import requests
import time
import random
from termcolor import colored
from typing import List, Dict, Optional

from models import Tool


class MovieAPIToolImpl:
    """Implementation for interacting with TMDB and OMDB APIs."""

    def __init__(self):
        self.tmdb_api_key = os.getenv("TMDB_API_KEY")
        self.omdb_api_key = os.getenv("OMDB_API_KEY")
        if not self.tmdb_api_key:
            print(
                colored(
                    "[MovieAPIToolImpl] Warning: TMDB_API_KEY not found in environment variables.",
                    "red",
                )
            )
        if not self.omdb_api_key:
            print(
                colored(
                    "[MovieAPIToolImpl] Warning: OMDB_API_KEY not found. OMDB failover will be disabled.",
                    "yellow",
                )
            )

    # --- Helper Methods for Search (Improved with retries and failover) ---
    def _search_tmdb(self, query: str) -> List[Dict]:
        """Internal method to search movies on TMDB."""
        if not self.tmdb_api_key:
            return []
        url = "https://api.themoviedb.org/3/search/movie"
        params = {
            "api_key": self.tmdb_api_key,
            "query": query,
            "include_adult": "false",
        }
        max_retries = 2
        backoff_factor = 1
        for attempt in range(1, max_retries + 1):
            try:
                # print(colored(f"[TMDB Search] Attempting search for '{query}' (Attempt {attempt}/{max_retries})", "cyan"))
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                results = data.get("results", [])
                if results:
                    processed_results = []
                    for movie in results:
                        processed_movie = {
                            "id": movie.get("id"),
                            "title": movie.get("title", "Unknown Title"),
                            "year": (
                                movie.get("release_date", "")[:4]
                                if movie.get("release_date")
                                else "Unknown Year"
                            ),
                            "rating": (
                                f"{movie.get('vote_average', 'N/A')}/10"
                                if movie.get("vote_average") is not None
                                else "N/A"
                            ),
                            "plot": movie.get("overview", "No plot summary available."),
                            "poster_path": movie.get("poster_path", ""),
                            "tmdb_id": movie.get("id"),
                            "imdb_id": movie.get("imdb_id", ""),
                            "source": "TMDb",
                        }
                        processed_results.append(processed_movie)
                    return processed_results
                else:
                    return []
            except requests.exceptions.Timeout:
                print(
                    colored(
                        f"[TMDB Search] Timeout error on attempt {attempt} for '{query}'.",
                        "red",
                    )
                )
            except requests.exceptions.RequestException as e:
                print(
                    colored(
                        f"[TMDB Search] Request error on attempt {attempt} for '{query}': {e}",
                        "red",
                    )
                )
            except Exception as e:
                print(
                    colored(
                        f"[TMDB Search] Unexpected error on attempt {attempt} for '{query}': {e}",
                        "red",
                    )
                )
                break
            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(
                    0, 1
                )
                time.sleep(sleep_time)
        print(
            colored(
                f"[MovieAPIToolImpl] TMDB search failed for '{query}' after {max_retries} attempts.",
                "red",
            )
        )
        return []

    def _search_omdb(self, query: str) -> List[Dict]:
        """Internal method to search movies on OMDB."""
        if not self.omdb_api_key:
            print(
                colored(
                    "[MovieAPIToolImpl] OMDB API key not configured for search failover.",
                    "red",
                )
            )
            return []
        omdb_url = "http://www.omdbapi.com/"
        omdb_params = {"apikey": self.omdb_api_key, "s": query, "type": "movie"}
        try:
            # print(colored(f"[OMDB Search] Attempting search for '{query}' via OMDB.", "cyan"))
            omdb_response = requests.get(omdb_url, params=omdb_params, timeout=10)
            omdb_response.raise_for_status()
            omdb_data = omdb_response.json()
            if omdb_data.get("Response") == "True":
                search_results = omdb_data.get("Search", [])[:10]
                processed_results = []
                for movie in search_results:
                    imdb_id = movie.get("imdbID", "")
                    movie_id = (
                        abs(hash(imdb_id)) % (2**31)
                        if imdb_id
                        else abs(hash(movie.get("Title", "Unknown"))) % (2**31)
                    )
                    processed_movie = {
                        "id": movie_id,
                        "title": movie.get("Title", "Unknown Title"),
                        "year": movie.get("Year", "Unknown Year"),
                        "rating": movie.get("imdbRating", "N/A"),
                        "plot": (
                            movie.get("Plot", "Plot not available in search results.")[
                                :200
                            ]
                            + "..."
                            if movie.get("Plot")
                            else "N/A"
                        ),
                        "poster_path": movie.get("Poster", ""),
                        "tmdb_id": None,
                        "imdb_id": imdb_id,
                        "source": "OMDb",
                    }
                    processed_results.append(processed_movie)
                return processed_results
            else:
                error_msg = omdb_data.get("Error", "Unknown error")
                print(
                    colored(
                        f"[MovieAPIToolImpl] OMDB search returned error for '{query}': {error_msg}",
                        "red",
                    )
                )
                return []
        except requests.exceptions.RequestException as e:
            print(
                colored(
                    f"[MovieAPIToolImpl] OMDB API network error for '{query}': {e}",
                    "red",
                )
            )
            return []
        except Exception as e:
            print(
                colored(
                    f"[MovieAPIToolImpl] Unexpected error during OMDB search for '{query}': {e}",
                    "red",
                )
            )
            return []

    # --- Public Methods ---
    def search_movie(self, title: str) -> List[Dict]:
        """
        Search for movies by title.
        Tries TMDB first. If TMDB fails or returns no results, falls back to OMDB.
        Returns a list of movie dictionaries with a consistent structure including an 'id'.
        """
        print(colored(f"[MovieAPIToolImpl] Searching for movie: '{title}'", "cyan"))
        tmdb_results = self._search_tmdb(title)
        if tmdb_results:
            print(
                colored(
                    f"[MovieAPIToolImpl] Found {len(tmdb_results)} results via TMDB for '{title}'.",
                    "green",
                )
            )
            return tmdb_results
        if self.omdb_api_key:
            print(
                colored(
                    f"[MovieAPIToolImpl] TMDB search failed/empty for '{title}'. Trying OMDB failover...",
                    "yellow",
                )
            )
            omdb_results = self._search_omdb(title)
            if omdb_results:
                print(
                    colored(
                        f"[MovieAPIToolImpl] Found {len(omdb_results)} results via OMDB failover for '{title}'.",
                        "green",
                    )
                )
                return omdb_results
            else:
                print(
                    colored(
                        f"[MovieAPIToolImpl] OMDB failover also failed/empty for '{title}'.",
                        "red",
                    )
                )
        else:
            print(
                colored(
                    f"[MovieAPIToolImpl] TMDB failed for '{title}' and OMDB_API_KEY not configured for failover.",
                    "red",
                )
            )
        print(
            colored(
                f"[MovieAPIToolImpl] Search for '{title}' yielded no results from TMDB or OMDB.",
                "red",
            )
        )
        return []

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        """
        Get detailed information about a movie from TMDB using its TMDB ID.
        """
        if not self.tmdb_api_key:
            print(
                colored(
                    "[MovieAPIToolImpl] TMDB API key not configured for get_movie_details.",
                    "red",
                )
            )
            return None
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            "api_key": self.tmdb_api_key,
            "language": "en-US",
            "append_to_response": "credits",
        }
        max_retries = 2
        backoff_factor = 1
        for attempt in range(1, max_retries + 1):
            try:
                # print(colored(f"[TMDB Details] Attempting to get details for movie ID {movie_id} (Attempt {attempt}/{max_retries})", "cyan"))
                response = requests.get(url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                director = "N/A"
                credits_data = data.get("credits", {})
                if credits_data:
                    crew = credits_data.get("crew", [])
                    for member in crew:
                        if member.get("job", "").lower() == "director":
                            director = member.get("name", "N/A")
                            break
                cast_list = []
                if credits_data:
                    cast_data = credits_data.get("cast", [])
                    for actor in cast_data[:5]:
                        cast_list.append(actor.get("name", "Unknown Actor"))
                runtime_minutes = data.get("runtime", 0)
                runtime_str = (
                    f"{runtime_minutes} min"
                    if runtime_minutes and runtime_minutes > 0
                    else "N/A"
                )
                genres_data = data.get("genres", [])
                genre_names = [
                    genre.get("name", "Unknown Genre") for genre in genres_data
                ]
                movie_details = {
                    "id": data.get("id"),
                    "title": data.get("title", "Unknown Title"),
                    "year": (
                        data.get("release_date", "")[:4]
                        if data.get("release_date")
                        else "Unknown Year"
                    ),
                    "rating": (
                        f"{data.get('vote_average', 'N/A')}/10"
                        if data.get("vote_average") is not None
                        else "N/A"
                    ),
                    "plot": data.get("overview", "No detailed plot summary available."),
                    "poster_path": data.get("poster_path", ""),
                    "genres": genre_names,
                    "cast": cast_list,
                    "director": director,
                    "runtime": runtime_str,
                    "tmdb_id": data.get("id"),
                    "imdb_id": data.get("imdb_id", ""),
                    "source": "TMDb",
                }
                print(
                    colored(
                        f"[MovieAPIToolImpl] Successfully fetched details for movie ID {movie_id}.",
                        "green",
                    )
                )
                return movie_details
            except requests.exceptions.Timeout:
                print(
                    colored(
                        f"[TMDB Details] Timeout error on attempt {attempt} for movie ID {movie_id}.",
                        "red",
                    )
                )
            except requests.exceptions.RequestException as e:
                print(
                    colored(
                        f"[TMDB Details] Request error on attempt {attempt} for movie ID {movie_id}: {e}",
                        "red",
                    )
                )
            except Exception as e:
                print(
                    colored(
                        f"[TMDB Details] Unexpected error on attempt {attempt} for movie ID {movie_id}: {e}",
                        "red",
                    )
                )
            if attempt < max_retries:
                sleep_time = backoff_factor * (2 ** (attempt - 1)) + random.uniform(
                    0, 1
                )
                time.sleep(sleep_time)
        print(
            colored(
                f"[MovieAPIToolImpl] Failed to get details for movie ID {movie_id} from TMDB after {max_retries} attempts.",
                "red",
            )
        )
        return None

    def get_movie_rating(self, title: str) -> Optional[Dict]:
        """Get movie rating, trying TMDB first, then OMDB as fallback."""
        tmdb_results = self.search_movie(
            title
        )  # This includes TMDB search and OMDB failover
        if tmdb_results:
            movie_data = tmdb_results[0]  # Already processed with ID
            return movie_data
        # Final OMDB fallback for specific rating lookup if search failed completely
        if self.omdb_api_key:
            print(
                colored(
                    f"[MovieAPIToolImpl] Final OMDB fallback for rating lookup of '{title}'.",
                    "yellow",
                )
            )
            omdb_url = "http://www.omdbapi.com/"
            omdb_params = {"t": title, "apikey": self.omdb_api_key}
            try:
                omdb_response = requests.get(omdb_url, params=omdb_params, timeout=10)
                omdb_response.raise_for_status()
                omdb_data = omdb_response.json()
                if omdb_data.get("Response") == "True":
                    imdb_id = omdb_data.get("imdbID", "")
                    movie_id = (
                        abs(hash(imdb_id)) % (2**31)
                        if imdb_id
                        else abs(hash(omdb_data.get("Title", "Unknown"))) % (2**31)
                    )
                    return {
                        "id": movie_id,
                        "title": omdb_data.get("Title"),
                        "year": omdb_data.get("Year", "Unknown"),
                        "rating": omdb_data.get("imdbRating", "N/A"),
                        "plot": omdb_data.get("Plot", "No plot summary available."),
                        "poster_path": omdb_data.get("Poster", ""),
                        "imdb_id": imdb_id,
                        "source": "OMDb",
                    }
                else:
                    print(
                        colored(
                            f"Movie not found on OMDB (final rating lookup): {omdb_data.get('Error', 'Unknown error')}",
                            "yellow",
                        )
                    )
            except requests.exceptions.RequestException as e:
                print(
                    colored(f"OMDB API network error (final rating lookup): {e}", "red")
                )
        return None

    def search_person(self, name: str) -> List[Dict]:
        """Search for a person by name on TMDB."""
        if not self.tmdb_api_key:
            return []
        url = "https://api.themoviedb.org/3/search/person"
        params = {"api_key": self.tmdb_api_key, "query": name, "include_adult": "false"}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("results", [])
        except requests.exceptions.RequestException as e:
            print(colored(f"TMDB Person Search API network error: {e}", "red"))
            return []

    def get_person_details(self, person_id: int) -> Optional[Dict]:
        """Get detailed information about a person from TMDB."""
        if not self.tmdb_api_key:
            return None
        url = f"https://api.themoviedb.org/3/person/{person_id}"
        params = {"api_key": self.tmdb_api_key}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(colored(f"TMDB Person Details API network error: {e}", "red"))
            return None

    def get_person_movie_credits(self, person_id: int) -> List[Dict]:
        """Get a list of movies the person is known for (acting roles) from TMDB."""
        if not self.tmdb_api_key:
            return []
        url = f"https://api.themoviedb.org/3/person/{person_id}/movie_credits"
        params = {"api_key": self.tmdb_api_key}
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            cast_credits = [
                credit for credit in data.get("cast", []) if credit.get("title")
            ]
            return cast_credits
        except requests.exceptions.RequestException as e:
            print(colored(f"TMDB Person Credits API network error: {e}", "red"))
            return []


class MovieAPITool(Tool):
    """Tool wrapper for MovieAPIToolImpl."""

    def __init__(self):
        super().__init__("MovieAPITool")
        self.api: Optional[MovieAPIToolImpl] = None

    def initialize(self) -> bool:
        try:
            self.api = MovieAPIToolImpl()
            return self.is_available()
        except Exception as e:
            print(colored(f"[{self.name}] Initialization error: {e}", "red"))
            return False

    def is_available(self) -> bool:
        return self.api is not None and self.api.tmdb_api_key is not None

    def search_movie(self, title: str) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.search_movie(title)

    def get_movie_details(self, movie_id: int) -> Optional[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.get_movie_details(movie_id)

    def get_movie_rating(self, title: str) -> Optional[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.get_movie_rating(title)

    def search_person(self, name: str) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.search_person(name)

    def get_person_details(self, person_id: int) -> Optional[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.get_person_details(person_id)

    def get_person_movie_credits(self, person_id: int) -> List[Dict]:
        if not self.is_available():
            raise RuntimeError("Tool not available")
        return self.api.get_person_movie_credits(person_id)
