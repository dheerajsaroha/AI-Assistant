import streamlit as st

from src.document_loader import load_documents
from src.rag_pipeline import RAGPipeline


# ================================================================
# PAGE CONFIG
# ================================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ================================================================
# CUSTOM CSS
# ================================================================

st.markdown(
    """
    <style>

    /* ------------------------------------------------------------
       Global
    ------------------------------------------------------------ */

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Reduce Streamlit default spacing */
    div[data-testid="stVerticalBlock"] {
        gap: 0.6rem;
    }

    /* ------------------------------------------------------------
       Header
    ------------------------------------------------------------ */

    .app-header {
        margin-bottom: 2rem;
    }

    .app-title {
        font-size: 2.35rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #8b8f98;
        margin-bottom: 0;
    }

    /* ------------------------------------------------------------
       Section headings
    ------------------------------------------------------------ */

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa0aa;
        margin-top: 1.4rem;
        margin-bottom: 0.55rem;
    }

    /* ------------------------------------------------------------
       Status
    ------------------------------------------------------------ */

    .status-text {
        font-size: 0.88rem;
        color: #a6aab2;
        margin-top: 0.15rem;
        margin-bottom: 0.5rem;
    }

    /* ------------------------------------------------------------
       Answer
    ------------------------------------------------------------ */

    .answer-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa0aa;
        margin-top: 1.6rem;
        margin-bottom: 0.65rem;
    }

    .answer-content {
        padding: 1.15rem 1.25rem;
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        line-height: 1.7;
    }

    /* ------------------------------------------------------------
       Sources
    ------------------------------------------------------------ */

    .sources-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9aa0aa;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* ------------------------------------------------------------
       Buttons
    ------------------------------------------------------------ */

    div.stButton > button {
        border-radius: 9px;
        min-height: 2.7rem;
        font-weight: 600;
    }

    /* ------------------------------------------------------------
       File uploader
    ------------------------------------------------------------ */

    section[data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
    }

    /* ------------------------------------------------------------
       Text area
    ------------------------------------------------------------ */

    textarea {
        border-radius: 10px !important;
    }

    /* ------------------------------------------------------------
       Expander
    ------------------------------------------------------------ */

    div[data-testid="stExpander"] {
        border-radius: 10px;
    }

    /* ------------------------------------------------------------
       Hide unnecessary Streamlit decoration
    ------------------------------------------------------------ */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }
    .answer-label {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #9aa0aa;
    margin-top: 1.6rem;
    margin-bottom: 0.65rem;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# SESSION STATE
# ================================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = []


# ================================================================
# HEADER
# ================================================================

st.markdown(
    """
    <div class="app-header">
        <div class="app-title">AI Document Assistant</div>
        <div class="app-subtitle">
            Ask questions and find answers across your documents.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ================================================================
# DOCUMENTS
# ================================================================

st.markdown(
    '<div class="section-label">Documents</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload documents",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "csv",
        "pptx",
        "xlsx",
    ],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


# ================================================================
# BUILD KNOWLEDGE BASE
# ================================================================

if uploaded_files:

    current_filenames = [
        file.name
        for file in uploaded_files
    ]

    previous_filenames = (
        st.session_state.uploaded_documents
    )

    files_changed = (
        current_filenames != previous_filenames
    )

    if files_changed:

        st.caption(
            f"{len(uploaded_files)} document(s) selected"
        )

        if st.button(
            "Build Knowledge Base",
            type="primary",
            use_container_width=True,
        ):

            with st.spinner(
                "Processing documents..."
            ):

                try:

                    all_documents = []

                    for uploaded_file in uploaded_files:

                        documents = load_documents(
                            uploaded_file
                        )

                        all_documents.extend(
                            documents
                        )

                    if not all_documents:
                        st.error(
                            "No readable content was found "
                            "in the uploaded documents."
                        )
                        st.stop()

                    pipeline = RAGPipeline()

                    pipeline.build_from_documents(
                        all_documents
                    )

                    st.session_state.pipeline = (
                        pipeline
                    )

                    st.session_state.uploaded_documents = (
                        current_filenames
                    )

                    st.success(
                        "Knowledge base created successfully."
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        "The documents could not be processed. "
                        "Please check the uploaded files and try again."
                    )
                    st.code(
                        str(error)
                    )


# ================================================================
# KNOWLEDGE BASE STATUS
# ================================================================

if st.session_state.pipeline is not None:

    pipeline = st.session_state.pipeline

    document_count = len(
        st.session_state.uploaded_documents
    )

    chunk_count = len(
        pipeline.chunks
    )

    st.markdown(
        f"""
        <div class="status-text">
            <strong>{document_count}</strong>
            document(s)
            &nbsp;•&nbsp;
            <strong>{chunk_count}</strong>
            indexed sections
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander(
        "View documents"
    ):

        for filename in (
            st.session_state.uploaded_documents
        ):
            st.write(
                f"📄 {filename}"
            )

    if st.button(
        "Clear Knowledge Base",
        use_container_width=True,
    ):

        st.session_state.pipeline = None
        st.session_state.uploaded_documents = []

        st.rerun()


# ================================================================
# QUESTION
# ================================================================

st.markdown(
    '<div class="section-label">Ask</div>',
    unsafe_allow_html=True,
)

query = st.text_area(
    "Question",
    placeholder="Ask a question about your documents...",
    height=80,
    label_visibility="collapsed",
)

ask_button = st.button(
    "Ask",
    type="primary",
    use_container_width=True,
)


# ================================================================
# ANSWER
# ================================================================

if ask_button:

    if st.session_state.pipeline is None:

        st.warning(
            "Build a knowledge base before asking a question."
        )

    elif not query.strip():

        st.warning(
            "Enter a question first."
        )

    else:

        with st.spinner(
            "Searching your documents..."
        ):

            try:

                result = (
                    st.session_state.pipeline.ask(
                        query=query,
                        k=5,
                        candidate_k=10,
                    )
                )

                # ------------------------------------------------
                # Answer
                # ------------------------------------------------

                st.markdown(
                    '<div class="answer-label">Answer</div>',
                    unsafe_allow_html=True,
                )

                with st.container(border=True):
                    st.markdown(
                        result["answer"]
    )

                # ------------------------------------------------
                # Sources
                # ------------------------------------------------

                sources = result.get(
                    "sources",
                    [],
                )

                if sources:

                    st.markdown(
                        '<div class="sources-label">Sources</div>',
                        unsafe_allow_html=True,
                    )

                    displayed_sources = set()

                    for source in sources:

                        document_name = source.get(
                            "document_name",
                            "Unknown document",
                        )

                        page = source.get(
                            "page",
                            None,
                        )

                        if isinstance(page, int):
                            display_page = page + 1
                        else:
                            display_page = page or "Unknown"

                        content = source.get(
                            "content",
                            "",
                        )

                        source_key = (
                            document_name,
                            page,
                        )

                        if source_key in displayed_sources:
                            continue

                        displayed_sources.add(
                            source_key
                        )

                        with st.expander(
                            f"📄 {document_name}  ·  Page {display_page}"
                        ):

                            st.write(
                                content
                            )

            except Exception:
                st.error(
                    "We couldn't answer that question. "
                    "Please try again."
                )