from PyPDF2 import PdfReader
import streamlit as st
from transformers import DistilBertForQuestionAnswering, DistilBertTokenizer
import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import torch.nn.functional as F

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

import re


def calculate_confidence(question, context, answer, model, tokenizer, max_length=512):
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = inputs["input_ids"].tolist()[0]
    
    # Find the start and end of the answer in the input_ids
    answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
    start_idx = None
    for i in range(len(input_ids) - len(answer_tokens) + 1):
        if input_ids[i:i+len(answer_tokens)] == answer_tokens:
            start_idx = i
            break
    
    if start_idx is None:
        return 0  # Answer not found in context
    
    end_idx = start_idx + len(answer_tokens) - 1
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    start_scores = outputs.start_logits[0]
    end_scores = outputs.end_logits[0]
    
    start_score = start_scores[start_idx].item()
    end_score = end_scores[end_idx].item()
    
    return (start_score + end_score) / 2

def clean_answer(answer):
    # Remove leading/trailing whitespace
    answer = answer.strip()
    
    # Remove trailing number and any following text
    answer = re.sub(r'\s+\d+.*$', '', answer)
    
    # Remove trailing punctuation
    answer = re.sub(r'[.,;:!?]$', '', answer)
    
    return answer


def clean_answer_with_confidence(question, context, answer, initial_confidence, model, tokenizer):
    best_answer = answer
    best_confidence = initial_confidence
    cleaned_answer = clean_answer(answer)
    print(f"Initial Answer: {answer} - Cleaned Answer: {cleaned_answer}")
    cleaned_confidence = calculate_confidence(question, context, cleaned_answer, model, tokenizer)
    
    if cleaned_confidence > initial_confidence:
        return cleaned_answer, cleaned_confidence
    else:
         # Iteratively remove trailing characters
        while len(answer) > 1:
            # Remove last character
            answer = answer[:-1].strip()
            
            # Remove trailing punctuation and numbers
            answer = re.sub(r'[.,;:!?]$', '', answer)
            answer = re.sub(r'\s+\d+$', '', answer)
            
            if answer == best_answer:
                continue  # Skip if the answer hasn't changed
            
            confidence = calculate_confidence(question, context, answer, model, tokenizer)
            
            if confidence > best_confidence:
                best_answer = answer
                best_confidence = confidence
            else:
                # If confidence didn't improve, stop iterating
                break
    
        return best_answer, best_confidence
    

def answer_question(question, context, model, tokenizer, max_length=512, stride=128):
    best_answer = ""
    best_confidence = float('-inf')
    
    # Split context into chunks with overlap
    for i in range(0, len(context), stride):
        chunk = context[i:i+max_length]
        
        inputs = tokenizer.encode_plus(question, chunk, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = inputs["input_ids"].tolist()[0]
        
        if len(input_ids) == 0:
            continue  # Skip empty token lists
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        start_probs = F.softmax(outputs.start_logits, dim=-1)
        end_probs = F.softmax(outputs.end_logits, dim=-1)
        
        # Find the best start and end indices
        start_idx = torch.argmax(start_probs)
        end_idx = torch.argmax(end_probs) + 1
        
        # Extract the answer
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[start_idx:end_idx]))
        
        # Calculate confidence using the improved method
        confidence = calculate_confidence(question, chunk, answer, model, tokenizer, max_length)
        
        if confidence > best_confidence and len(answer.strip()) > 0:
            best_confidence = confidence
            best_answer = answer
    
    print(f"Answer: {best_answer} - Confidence: {best_confidence}")
    return best_answer, best_confidence


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
            initial_answer, initial_confidence = answer_question(query, context, model, tokenizer)  
            cleaned_answer, confidence = clean_answer_with_confidence(query, context, initial_answer, initial_confidence, model, tokenizer)
            print(f"Raw Answer: {initial_answer} - Answer: {cleaned_answer}")      
            st.write(f"Answer: {cleaned_answer}")
            st.write(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()