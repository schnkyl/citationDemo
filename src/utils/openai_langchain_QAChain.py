### USES THIS DOCUMENT AS REFERENCE: https://python.langchain.com/v0.2/docs/tutorials/pdf_qa/

import time
from typing_extensions import Annotated, TypedDict, Optional, List

from dotenv import load_dotenv  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain.chat_models import init_chat_model # type: ignore
from langchain_core.documents import Document # type: ignore
from langgraph.graph import START, StateGraph # type: ignore

load_dotenv() # Only need when bypassing main

llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")

class CitationPageLocation(TypedDict):
    cited_text: Annotated[str, ..., "The text that is cited"]
    document_index: Annotated[int, ..., "Index of the document"]
    document_title: Annotated[str, ..., "Title of the document"]
    end_page_number: Annotated[int, ..., "End page number of the citation"]
    start_page_number: Annotated[int, ..., "Start page number of the citation"]

class CitedAnswer(TypedDict):
    citations: Optional[List[CitationPageLocation]]
    text: Annotated[str, ..., "Explaination of all cited text"]
    type: Annotated[str, ..., "The type of the block"]

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\nHere are the documents: "
    "{context}"
)


loader = PyPDFLoader('/Users/kyle/Documents/BreezeRFP/citationDemo/src/Boston - Mobile App Development RFP.pdf')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,  # Recommended to have some overlap between chunks
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

chunks = text_splitter.split_documents(docs)

vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
        
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

class OpenAI_LangChain:
    def get_citations(self, pdf_path, question):
        

        # We will do everything above on page load, not effecting the customer
        start_time = time.time()
        try:
            question_answer_chain = create_stuff_documents_chain(self.llm, prompt)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": question})
        except Exception as e:
            print(f"An error occurred: {e}")
            response = None
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Time taken: {elapsed_time} seconds")
        print(response)
        return response

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: CitedAnswer


# Define application steps
def retrieve(state: State):
    retrieved_docs = retriever.invoke(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    structured_llm = llm.with_structured_output(CitedAnswer)
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = structured_llm.invoke(messages)
    return {"answer": response}

# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

result = graph.invoke({"question": "What are the deliverables?"})

sources = [doc.metadata["source"] for doc in result["context"]]
print(f"Sources: {sources}\n\n")
print(f'Answer: {result["answer"]}')