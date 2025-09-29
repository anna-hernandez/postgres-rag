import streamlit as st
from main import rag


def get_response(prompt):
    # Simulate a response from an AI model
    return f"Response to: {prompt}"


if __name__ == "__main__":
    st.title("RAG with Postgres and pgvector")
    st.write(
        "This is a simple app demonstrating Retrieval-Augmented Generation (RAG) using Postgres and pgvector."
    )

    # sidebar for query type selection
    # default to hybrid
    query_type = st.radio(
        label="Type of query",
        options=["sql", "keyword", "semantic", "hybrid"],
        index=3,
    )

    if prompt := st.chat_input("What do you want to know?"):
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []
        else:
            for chat in st.session_state.chat_history:
                with st.chat_message(chat["role"]):
                    st.markdown(chat["content"])

        st.session_state.chat_history.append(
            {"role": "user", "content": prompt},
        )
        # display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # display assistant response in chat message container
        with st.chat_message("assistant"):

            response = rag(
                create_db=False,
                query=prompt,
                query_type=query_type,
            )
            st.markdown(response)  # call the rag function from main.py

            st.session_state.chat_history.append(
                {"role": "assistant", "content": response},
            )
