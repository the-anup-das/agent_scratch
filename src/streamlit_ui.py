from orchestrator import UniversalAssistantSystem


def run_streamlit_ui():
    """
    Run the Agentic Movie Recommendation System in Streamlit UI mode.
    """
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed. Run `pip install streamlit` to launch the UI."
        )
        return

    st.set_page_config(page_title="🎬 Agentic Movie Assistant", page_icon="🎥")
    st.title("🎬 Agentic Movie Assistant")
    st.markdown(
        "Ask me anything about movies, actors, genres, hidden gems, or recommendations.\n\n"
        "Enhanced with hybrid search, refined entity extraction, TMDB/OMDB fallback, rating-based ranking."
    )

    # Initialize assistant and chat history
    if "assistant" not in st.session_state:
        st.session_state.assistant = UniversalAssistantSystem()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.experimental_rerun()

    # Display demo suggestions
    with st.expander("💡 Try one of these questions", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎬 Thriller like Inception"):
                st.session_state.messages.append(
                    {"role": "user", "content": "Recommend a thriller like Inception"}
                )
                st.experimental_rerun()
            if st.button("⭐ Best Nolan movies"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "List the best Christopher Nolan movies ranked by rating",
                    }
                )
                st.experimental_rerun()
            if st.button("👩‍🎤 Emma Stone + Gosling"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "Give me movies with Emma Stone and Ryan Gosling",
                    }
                )
                st.experimental_rerun()
        with col2:
            if st.button("📊 Compare ratings for Interstellar"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "Compare IMDb and TMDB ratings for Interstellar",
                    }
                )
                st.experimental_rerun()
            if st.button("🕵️‍♂️ Underrated thrillers"):
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": "What are some underrated thrillers from the 2010s?",
                    }
                )
                st.experimental_rerun()
            if st.button("🏆 Best Picture 2023"):
                st.session_state.messages.append(
                    {"role": "user", "content": "Who won Best Picture in 2023?"}
                )
                st.experimental_rerun()

    # Display message history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("What would you like to know?")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Thinking..."):
            try:
                response = st.session_state.assistant.chat(user_input)
            except Exception as e:
                response = f"⚠️ An error occurred: `{e}`"

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
