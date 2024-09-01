from PyPDF2 import PdfReader
import streamlit as st
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def answer_question(question, context, model, tokenizer, max_length=512):
    # Split the context into chunks of max_length
    chunks = []
    for i in range(0, len(context), max_length):
        chunk = context[i:i+max_length]
        chunks.append(chunk)
    
    best_answer = ""
    best_score = float('-inf')

    for chunk in chunks:
        inputs = tokenizer.encode_plus(question, chunk, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]

        outputs = model(**inputs)
        answer_start_scores = outputs.start_logits
        answer_end_scores = outputs.end_logits

        answer_start = torch.argmax(answer_start_scores)
        answer_end = torch.argmax(answer_end_scores) + 1

        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end]))
        
        # Calculate the score for this answer
        score = torch.max(answer_start_scores) + torch.max(answer_end_scores)
        
        if score > best_score:
            best_score = score
            best_answer = answer

    return best_answer

# process text from pdf
def process_text(text):
    # split the text into chunks using langchain
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # convert the chunks of text into embeddings to form a knowledge base
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base

def main():
    st.title("Chat with the docs")
    pdf = st.file_uploader("Upload your PDF File", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        # store the pdf text in a var
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # create a knowledge base object
        knowledgeBase = process_text(text)
        
        query = st.text_input('Ask question to PDF...')
        cancel_button = st.button('Cancel')
        
        if cancel_button:
            st.stop()
        
        if query:
            docs = knowledgeBase.similarity_search(query)

            model_name = "distilbert-base-uncased-distilled-squad"
            model = DistilBertForQuestionAnswering.from_pretrained(model_name)
            tokenizer = DistilBertTokenizer.from_pretrained(model_name)

            # Combine all document texts
            context = " ".join([doc.page_content for doc in docs])

            # Get the answer
            answer = answer_question(query, context, model, tokenizer)
            
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()