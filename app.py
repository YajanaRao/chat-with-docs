from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import langchain
from transformers import AutoModelForCausalLM, AutoTokenizer


import torch

langchain.verbose = False


def answer_question(question, context, model, tokenizer, max_length=512, max_new_tokens=50):
    # Prepare the input text
    input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    # Encode the input text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Generate the answer
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids, 
            max_new_tokens=max_new_tokens,  # Use max_new_tokens instead of max_length
            num_return_sequences=1, 
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
    
    # Decode the generated answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part
    answer = answer.split("Answer:")[-1].strip()
    return answer

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

            # Load pre-trained model and tokenizer
            model_name = "facebook/opt-1.3b"
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)

              # Combine all document texts
            context = " ".join([doc.page_content for doc in docs])

            # Get the answer
            answer = answer_question(query, context, model, tokenizer, max_length=512, max_new_tokens=50)
          
            st.write(f"Answer: {answer}")

if __name__ == "__main__":
    main()