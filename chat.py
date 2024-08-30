from PyPDF2 import PdfReader
import streamlit as st
from script import query_rag



def main():
    st.title("Chat with my PDF")
    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        text = "".join([page.extract_text() for page in pdf_reader.pages])

        print("Processing text...")
        query = st.text_input('Ask a question to your PDF...')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            print("Processing query...")
            summary = query_rag(query, text)
            st.write(f"Summary: {summary}")

if __name__ == "__main__":
    main()
