import time
from typing_extensions import Annotated, TypedDict, Optional, List

from dotenv import load_dotenv  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore
from langchain.chat_models import init_chat_model  # type: ignore
from langchain_core.documents import Document  # type: ignore
from langgraph.graph import START, StateGraph  # type: ignore

load_dotenv()  # Only need when bypassing main

llm = init_chat_model("gpt-4o-mini-2024-07-18", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

Q1 = "What are the deliverables?"
Q2 = "Give me a list of key dates."
Q3 = "Is there anything that is not straight forward or normal in the context of RFPs?"


class CitationPageLocation(TypedDict):
    cited_text: Annotated[str, ..., "The text that is cited"]
    document_id: Annotated[
        str, ..., "uuid of the document where the cited_text came from"
    ]


class CitedAnswer(TypedDict):
    citations: Optional[List[CitationPageLocation]]
    text: Annotated[
        str,
        ...,
        "Summary of all cited text to, that gives an answer to the users question",
    ]


system_prompt = (
    "You are an assistant for question-answering tasks about PDF documents. "
    "Use as many of the following pieces of retrieved context to answer "
    "the question. Make sure any infomration that may be helpful to the person asking the question is shared. "
    "If you don't know the answer, say that you don't know. Keep the answer concise. "
    "Make sure to follow the proper structure in your response."
    "\n\nHere are the documents: "
    "{context}"
)

loader = PyPDFLoader(
    "/Users/kyle/Documents/BreezeRFP/citationDemo/src/Boston - Mobile App Development RFP.pdf"
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
    keep_separator=False,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
    ]
)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: CitedAnswer


def retrieve(state: State):
    chunks = text_splitter.split_documents(docs)

    global vectorstore
    vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    retrieved_docs = retriever.invoke(state["question"], k=20)
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    structured_llm = llm.with_structured_output(CitedAnswer)
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = structured_llm.invoke(messages)
    return {"answer": response}


start_time = time.time()
try:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    result = graph.invoke({"question": Q2})

    answer_text = str(result["answer"]['text'])
    print(f'Answer: {answer_text}\n')
    for i, citation in enumerate(result["answer"]["citations"]):
        print(f'Cited text: {citation["cited_text"]}\n')
        similar_docs = vectorstore.similarity_search(str(citation["cited_text"]))
        print(f'similar_docs[0][page_content] {similar_docs[0]}\n\n\n==================================================================\n\n\n')

except Exception as e:
    print(f"An error occurred: {e}")
    response = None
end_time = time.time()
elapsed_time = round(end_time - start_time, 2)
print(f"Time taken: {elapsed_time} seconds")
