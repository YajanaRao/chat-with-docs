from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import FAISS
import streamlit as st
from ragatouille import RAGPretrainedModel
from transformers import AutoModelForCausalLM, AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer
from typing import Optional, Tuple, List
from PyPDF2 import PdfReader
import torch

# Assuming you have defined RAG_PROMPT_TEMPLATE somewhere
RAG_PROMPT_TEMPLATE = """
Answer:
{context}
"""

SUMMARY_PROMPT_TEMPLATE = """

{answer}

Summary:
"""

def generate_text(prompt, model, tokenizer, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids, 
                max_new_tokens=max_new_tokens, 
                num_return_sequences=1,
                do_sample=True,
                top_p=0.95,
                temperature=0.7
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except IndexError:
        return "I'm sorry, I couldn't generate a response. Could you please rephrase your question?"

def answer_with_rag(
    question: str,
    model,
    tokenizer,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[str]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Generate an answer
    print("=> Generating answer...")
    answer = generate_text(final_prompt, model, tokenizer, max_new_tokens=150)

    return answer, relevant_docs


def summarize_answer(answer, model, tokenizer):
    inputs = tokenizer("summarize: " + answer, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


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
    st.title("Chat with my PDF")
    pdf = st.file_uploader("Upload your PDF File", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        
        # store the pdf text in a var
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # create a knowledge base object
        knowledgeBase = process_text(text)

         # Load pre-trained model and tokenizer for QA
        qa_model_name = "facebook/opt-1.3b"
        qa_model = AutoModelForCausalLM.from_pretrained(qa_model_name)
        qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)

        # Load pre-trained model and tokenizer for summarization
        sum_model_name = "t5-base"  # You can also try "t5-small" or "t5-large"
        sum_model = T5ForConditionalGeneration.from_pretrained(sum_model_name)
        sum_tokenizer = T5Tokenizer.from_pretrained(sum_model_name)
       
        query = st.text_input('Ask question to PDF...')
        cancel_button = st.button('Cancel')

        if cancel_button:
            st.stop()

        if query:
            answer, relevant_docs = answer_with_rag(query, qa_model, qa_tokenizer, knowledgeBase)
            
            st.write(f"Full Answer: {answer}")
            
            if len(answer) > 50:  # Only summarize if the answer is long enough
                summary = summarize_answer(answer, sum_model, sum_tokenizer)
                st.write(f"Summary: {summary}")
            else:
                st.write("The answer is concise enough and doesn't need summarization.")
            
            st.write("Relevant documents:")
            for i, doc in enumerate(relevant_docs):
                st.write(f"Document {i + 1}:")
                st.write(doc)

if __name__ == "__main__":
    main()