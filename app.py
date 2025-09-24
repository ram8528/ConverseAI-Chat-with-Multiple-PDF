import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from datetime import datetime, timedelta
from htmlTemplates import css, bot_template, user_template
import base64

# Load environment variables from .env file
load_dotenv()

# Inject custom CSS for the input placeholder
def inject_custom_css():
    st.markdown(
        """
        <style>
        input::placeholder {
            color: black !important;
            opacity: 1 !important;  /* Ensures the placeholder is fully opaque */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to encode PDF directly in base64 for inline viewing
def encode_pdf_base64(pdf):
    base64_pdf = base64.b64encode(pdf.getvalue()).decode("utf-8")
    return f"data:application/pdf;base64,{base64_pdf}"

# Function to display PDF in an iframe (works better than base64 embedding)
def display_pdf_with_iframe(pdf_file):
    pdf_base64 = base64.b64encode(pdf_file.read()).decode("utf-8")
    pdf_url = f"data:application/pdf;base64,{pdf_base64}"
    iframe = f'<iframe src="{pdf_url}" width="100%" height="600px"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)

# Function to extract text with page and line numbers
def get_pdf_text(pdf_docs):
    pdf_data = []
    for pdf_index, pdf in enumerate(pdf_docs):
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            lines = page.extract_text().split("\n")
            for line_num, line in enumerate(lines, start=1):
                pdf_data.append({
                    "pdf_index": pdf_index,
                    "page": page_num,
                    "line": line_num,
                    "content": line
                })
    return pdf_data

# Function to split text into manageable chunks with references
def get_text_chunks(pdf_data):
    text_chunks = []
    chunk_content = ""
    chunk_references = []

    for ref in pdf_data:
        chunk_content += ref["content"] + "\n"
        chunk_references.append(f"Page {ref['page']}, Line {ref['line']}")

        if len(chunk_content) >= 1000:
            text_chunks.append({"content": chunk_content.strip(), "references": chunk_references})
            chunk_content = ""
            chunk_references = []

    if chunk_content:
        text_chunks.append({"content": chunk_content.strip(), "references": chunk_references})

    return text_chunks

# Function to create a vectorstore with references
def get_vectorstore(text_chunks_with_references):
    embeddings = OpenAIEmbeddings()
    texts = [chunk["content"] for chunk in text_chunks_with_references]
    metadata = [{"references": ", ".join(chunk["references"])} for chunk in text_chunks_with_references]
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadata)
    return vectorstore

# Function to create a conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and display page/line references
def handle_userinput():
    user_question = st.session_state.user_question

    if st.session_state.conversation is not None and user_question:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']
        last_response = response['chat_history'][-1].content

        negation_phrases = [
            "I don't have access",
            "I cannot access the document",
            "unable to provide specific content",
            "does not exist",
            "not able to retrieve",
        ]

        contains_all_negations = all(phrase.lower() in last_response.lower() for phrase in negation_phrases)
        contains_some_negations = any(phrase.lower() in last_response.lower() for phrase in negation_phrases)

        formatted_response = last_response

        if not contains_all_negations or contains_some_negations or 'source_documents' in response:
            sources = []
            for doc in response.get('source_documents', []):
                metadata = doc.metadata
                page_number = metadata.get('page_number')
                pdf_index = metadata.get('pdf_index', 0)

                if page_number is not None and pdf_index is not None:
                    try:
                        base64_pdf = encode_pdf_base64(st.session_state.uploaded_pdfs[pdf_index])
                        pdf_link = (
                            f"<a href='{base64_pdf}#page={page_number}' target='_blank' "
                            f"style='text-decoration:none;color:blue;'>Page {page_number}</a>"
                        )
                        sources.append(pdf_link)
                    except Exception as e:
                        sources.append(f"Error generating link for page {page_number}: {e}")

            if sources:
                formatted_response = f"{last_response}\n\n**References:**\n" + "\n".join(sources)
            else:
                formatted_response = f"{last_response}\n\n**References:**\nNo references found."
        elif contains_all_negations:
            formatted_response = f"{last_response}\n\n**References:**\nNo references found due to access restrictions."
        elif contains_some_negations:
            formatted_response = f"{last_response}\n\n**References:**\nSome information may not be available due to access limitations."

        current_date = datetime.now().strftime('%Y-%m-%d')
        if current_date not in st.session_state.questions_by_date:
            st.session_state.questions_by_date[current_date] = []

        st.session_state.questions_by_date[current_date].append({
            "user": user_question,
            "bot": formatted_response
        })

        st.session_state.user_question = ""

# Function to display chat history
def display_chat_history():
    today = datetime.now().strftime('%Y-%m-%d')
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    day_before_yesterday = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')

    if today in st.session_state.questions_by_date:
        st.sidebar.subheader("Today's History")
        for i, qa_pair in enumerate(st.session_state.questions_by_date[today]):
            if st.sidebar.button(f"Q{i + 1}: {qa_pair['user'][:30]}..."):
                st.session_state.selected_question = qa_pair["user"]
                st.session_state.chat_history = qa_pair["bot"]
                break

    if yesterday in st.session_state.questions_by_date:
        st.sidebar.subheader("Yesterday's History")
        for i, qa_pair in enumerate(st.session_state.questions_by_date[yesterday]):
            if st.sidebar.button(f"Q{i + 1}: {qa_pair['user'][:30]}..."):
                st.session_state.selected_question = qa_pair["user"]
                st.session_state.chat_history = qa_pair["bot"]
                break

    if day_before_yesterday in st.session_state.questions_by_date:
        st.sidebar.subheader("Day Before Yesterday's History")
        for i, qa_pair in enumerate(st.session_state.questions_by_date[day_before_yesterday]):
            if st.sidebar.button(f"Q{i + 1}: {qa_pair['user'][:30]}..."):
                st.session_state.selected_question = qa_pair["user"]
                st.session_state.chat_history = qa_pair["bot"]
                break

# Main app logic
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    inject_custom_css()

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "questions_by_date" not in st.session_state:
        st.session_state.questions_by_date = {}
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    if "selected_question" not in st.session_state:
        st.session_state.selected_question = ""
    if "uploaded_pdfs" not in st.session_state:
        st.session_state.uploaded_pdfs = []

    st.header("Ask Anything About Your Uploaded PDFs")

    with st.sidebar:
        st.title('✨ Converse AI 🗨️ \n Chat with multiple PDFs :books:')
        source_option = st.radio("Select Upload Source", ('Browser', 'SharePoint'))

        if source_option == 'Browser':
            st.subheader("Your documents")
            pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)

            if pdf_docs:
                st.session_state.uploaded_pdfs = pdf_docs

        elif source_option == 'SharePoint':
            st.subheader("Enter SharePoint details:")
            sharepoint_url = st.text_input("SharePoint Site URL")
            sharepoint_path = st.text_input("SharePoint Path")
            sharepoint_username = st.text_input("SharePoint Username")
            sharepoint_password = st.text_input("SharePoint Password", type="password")

        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = ""
                if source_option == 'Browser' and st.session_state.uploaded_pdfs:
                    raw_text = get_pdf_text(st.session_state.uploaded_pdfs)
                elif source_option == 'SharePoint':
                    st.write(f"Processing SharePoint PDF from {sharepoint_url}...")

                if raw_text:
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Processing completed!")
                else:
                    st.warning("No content found to process.")

    if st.session_state.questions_by_date:
        for date, qa_pairs in st.session_state.questions_by_date.items():
            for qa_pair in qa_pairs:
                st.write(user_template.replace("{{MSG}}", qa_pair["user"]), unsafe_allow_html=True)
                st.write(bot_template.replace("{{MSG}}", qa_pair["bot"]), unsafe_allow_html=True)

    if st.session_state.selected_question == "":
        st.text_input(
            label="",
            placeholder="Enter Your Query Prompt:",
            key="user_question",
            on_change=handle_userinput
        )
    else:
        st.write(user_template.replace("{{MSG}}", st.session_state.selected_question), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", st.session_state.chat_history), unsafe_allow_html=True)

    display_chat_history()

# Run the app
if __name__ == '__main__':
    main()
