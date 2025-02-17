### USES THIS DOCUMENT AS REFERENCE: https://python.langchain.com/docs/how_to/qa_citations

import time
from typing_extensions import Annotated, TypedDict, Optional, List
from dotenv import load_dotenv  # type: ignore

from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langgraph.graph import START, StateGraph  # type: ignore
from langchain.retrievers.document_compressors import EmbeddingsFilter  # type: ignore
from IPython.display import Image, display # type: ignore

load_dotenv()

llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

Q1 = "What are the deliverables?"
Q2 = "Give me a list of key dates."
Q3 = "Is there anything that is not straight forward or normal in the context of RFPs?"

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

loader = PyPDFLoader(
        "/Users/kyle/Documents/BreezeRFP/citationDemo/src/Boston - Mobile App Development RFP.pdf"
    )
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", " "],
    keep_separator=False,
)

# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: CitedAnswer

def retrieve(state: State):
    split_docs = splitter.split_documents(docs)
    compressor = EmbeddingsFilter(embeddings=embeddings, k=int(len(split_docs) * 0.2))
    stateful_docs = compressor.compress_documents(split_docs, state["question"])
    return {"context": stateful_docs}

def generate(state: State):
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{question}"),
        ]
    )
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # structured_llm = llm.with_structured_output(CitedAnswer)
    messages = prompt.invoke(
        {"question": state["question"], "context": docs_content}
    )
    response = llm.invoke(messages)
    return {"answer": response}



start_time = time.time()
try:
    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": "What are the deliverables?"})

    sources = [doc.metadata["source"] for doc in result["context"]]
    print(f'Answer: {result["answer"]}\n\n')
    print(len(result["context"]))
    print(f'Page_content: {result["context"][0].page_content}\n\n')
    print(f'Summary: {result["context"][0].metadata}\n\n')
except Exception as e:
    print(f"An error occurred: {e}")
    response = None
end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print(f"Time taken: {elapsed_time} seconds")


