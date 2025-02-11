import base64
import time
from typing_extensions import Annotated, TypedDict, Optional, List

from langchain.chat_models import init_chat_model  # type: ignore
from langchain_chroma import Chroma  # type: ignore
from langchain_openai import OpenAIEmbeddings  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain.chains import create_retrieval_chain  # type: ignore
from langchain_community.document_loaders import PyPDFLoader  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore


class CitationPageLocation(TypedDict):
    cited_text: Annotated[str, ..., "The text that is cited"]
    document_index: Annotated[int, ..., "Index of the document"]
    document_title: Annotated[str, ..., "Title of the document"]
    end_page_number: Annotated[int, ..., "End page number of the citation"]
    start_page_number: Annotated[int, ..., "Start page number of the citation"]
    type: Annotated[str, ..., "Type of the citation"]


class TextBlock(TypedDict):
    citations: Optional[List[CitationPageLocation]]
    text: Annotated[str, ..., "Explaination of the cited text"]
    type: Annotated[str, ..., "The type of the block"]


system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)


class Gemini:
    def __init__(self, api_key):
        llm = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        self.structured_llm = llm.with_structured_output(TextBlock)

    def get_citations(self, pdf_path, question):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        start_time = (
            time.time()
        )  # if we do this we will be able to do this before hand, not counting toward the time
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=splits, embedding=OpenAIEmbeddings()
            )
            retriever = vectorstore.as_retriever()

            combine_docs_chain = create_stuff_documents_chain(
                self.structured_llm, prompt
            )
            retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

            retrieval_chain.invoke({"input": question})
        except Exception as e:
            print(f"An error occurred: {e}")
            response = None
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Time taken: {elapsed_time} seconds")
        print(response)
        return response
