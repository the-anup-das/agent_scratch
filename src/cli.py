from orchestrator import UniversalAssistantSystem
from termcolor import colored


def run_cli():
    """Run the system in CLI mode"""
    print(
        colored(
            "🎬 Agentic Assistant System",
            "green",
            attrs=["bold"],
        )
    )
    print("=" * 85)
    print(
        colored(
            "Enhanced with hybrid search, refined entity extraction, Pydantic, OMDB/TMDB failover, rating-based sorting, and PERSON support.",
            "blue",
        )
    )
    print(colored("Type 'quit' or 'exit' to stop.", "yellow"))
    print("=" * 85)
    assistant = UniversalAssistantSystem()
    while True:
        user_input = input(colored("You: ", "cyan"))
        if user_input.strip().lower() in ["quit", "exit"]:
            print(colored("Goodbye!", "green"))
            break
        response = assistant.chat(user_input)
        print(colored(f"Assistant: {response}", "magenta"))
