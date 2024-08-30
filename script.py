
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama



PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str, context_text: str) -> str: 
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = Ollama(base_url='http://localhost:11434',
    model="llama3")
    response_text = model.invoke(prompt)

    print(response_text)
    return response_text

