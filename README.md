

# Agentic Assistant System - Build agent from scratch 

Build a modular agentic assistant from scratch for movies, ratings, and general knowledge.

## Overview

This Python system provides:
- **Movie recommendations** (hybrid semantic + TMDB API)
- **Movie ratings and info**
- **General knowledge answers**

## Features

- Hybrid search: semantic vector DB + TMDB API
- Entity extraction for personalized recommendations
- CLI and Streamlit web UI
- Modular agent architecture (general knowledge, movie search, person search)
- Extensible tools for LLM, vector DB, and movie APIs

## Setup

1. **Install dependencies**
   Uses [uv](https://github.com/astral-sh/uv) for fast, reproducible installs:
   ```cmd
   uv sync
   ```
   (Update dependencies and lockfile with `uv` as needed.)

2. **Set API keys**
   - [TMDB API Key](https://www.themoviedb.org/settings/api)
   - (Optional) [OpenAI API Key](https://platform.openai.com/)

   On Windows (CMD):
   ```cmd
   set TMDB_API_KEY=your_tmdb_key
   set OPENAI_API_KEY=your_openai_key
   ```

3. **Run in CLI mode**
   ```cmd
   python src/cli.py
   ```
   or
   ```cmd
   python src/main.py --mode cli
   ```

4. **Run in Streamlit UI**
   ```cmd
   streamlit run src/streamlit_ui.py
   ```

## Requirements

- Python 3.8+
- See `pyproject.toml` for dependencies

## File Structure

```
agent_scratch/
├── src/
│   ├── cli.py
│   ├── main.py
│   ├── orchestrator.py
│   ├── streamlit_ui.py
│   ├── agents/
│   │   ├── general_knowledge.py
│   │   ├── movie_search.py
│   │   ├── person_search.py
│   │   ├── query_analyzer.py
│   │   └── response_formatter.py
│   ├── db_index/
│   │   ├── wiki_movies_index_final.faiss
│   │   ├── wiki_movies_metadata_final.pkl
│   │   └── wiki_movies_texts_final.pkl
│   ├── models/
│   │   └── base.py
│   └── tools/
│       ├── general_knowledge.py
│       ├── llm_interface.py
│       ├── movie_api.py
│       └── vector_db.py
├── main.py
├── pyproject.toml
├── README.md
├── LICENSE
├── uv.lock
└── test.ipynb
```

## License

See [LICENSE](LICENSE).
