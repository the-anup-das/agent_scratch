from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from termcolor import colored


class QueryAnalysis(BaseModel):
    topic: str = Field(
        ..., description="Topic of the query: 'movie', 'person', 'general'"
    )
    intent: str = Field(
        ...,
        description="Intent: 'recommendation', 'specific_info', 'information', 'find_movies'",
    )
    specific_request: str = Field(
        default="", description="Specific movie title, person name, or query"
    )
    entities: Dict[str, Any] = Field(
        default_factory=dict, description="Extracted entities"
    )


class MovieData(BaseModel):
    id: int
    title: str
    year: Union[str, int]
    rating: str
    plot: str
    poster_path: str = ""
    genres: List[str] = []
    cast: List[str] = []
    director: str = ""
    runtime: str = ""
    tmdb_id: Optional[int] = None
    imdb_id: Optional[str] = None
    source: str = "Unknown"


class PersonData(BaseModel):
    id: int
    name: str
    biography: str
    birth_date: str = ""
    death_date: Optional[str] = None
    place_of_birth: str = ""
    profile_path: str = ""
    known_for_department: str = ""
    popularity: float = 0.0
    imdb_id: Optional[str] = None
    source: str = "Unknown"


class MovieRecommendationResult(BaseModel):
    topic: str = "movie"
    result: List[MovieData]


class PersonResult(BaseModel):
    person: PersonData
    known_for_movies: List[MovieData]


class PersonSearchResult(BaseModel):
    topic: str = "person"
    result: PersonResult


class GeneralKnowledgeResult(BaseModel):
    topic: str = "general"
    result: str


class FormattedResponse(BaseModel):
    final_response: str
    movie_recommendations: Optional[List[MovieData]] = None
    general_info: Optional[str] = None
    person_info: Optional[PersonResult] = None


class ProcessingContext(BaseModel):
    user_input: str
    timestamp: str
    analysis: Optional[QueryAnalysis] = None
    movie_result: Optional[MovieRecommendationResult] = None
    general_result: Optional[GeneralKnowledgeResult] = None
    person_result: Optional[PersonSearchResult] = None
    formatted_response: Optional[FormattedResponse] = None


# ==================== TOOL ARCHITECTURE ====================
class Tool(ABC):
    """Abstract base class for all tools."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the tool (e.g., load API keys, models). Return True on success."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the tool is properly initialized and available."""
        pass


class ToolManager:
    """Manages the registration, initialization, and access of tools."""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register_tool(self, tool: Tool):
        """Register a tool instance."""
        self.tools[tool.name] = tool

    def initialize_all(self):
        """Initialize all registered tools."""
        for name, tool in self.tools.items():
            print(colored(f"[ToolManager] Initializing tool: {name}", "blue"))
            success = tool.initialize()
            if not success:
                print(
                    colored(
                        f"[ToolManager] Warning: Initialization failed for tool: {name}",
                        "red",
                    )
                )

    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool instance by name IF it's available."""
        tool = self.tools.get(tool_name)
        if tool and tool.is_available():
            return tool

        # print(colored(f"[ToolManager] Tool '{tool_name}' not found or not available.", "yellow")) # Optional debug log


# ==================== BASE AGENT CLASS ====================
class Agent(ABC):
    """Base agent class"""

    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    @abstractmethod
    def execute(
        self,
        context: ProcessingContext,
        tool_manager: ToolManager,
        available_tools: List[Tool],
        required_tool_names: List[str],
    ) -> Dict[str, Any]:
        """Execute the agent's task. Receives pre-fetched available tools."""
        pass
