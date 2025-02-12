### Started using RetrievalQAWithSourcesChain: https://python.langchain.com/api_reference/langchain/chains/langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.html#langchain.chains.qa_with_sources.retrieval.RetrievalQAWithSourcesChain.combine_documents_chain

import time
from typing_extensions import Annotated, TypedDict, Optional, List

from langchain_chroma import Chroma  # type: ignore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
# from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain.chains import RetrievalQAWithSourcesChain  # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain  # type: ignore
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

class OpenAI_LangChain:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18"
        )
        # self.structured_llm = llm.with_structured_output(TextBlock)


    def get_citations(self, pdf_path, question):
        loader = PyPDFLoader('/Users/kyle/Documents/BreezeRFP/citationDemo/src'+ pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=OpenAIEmbeddings()
        )

        # We will do everything above on page load, not effecting the customer
        start_time = time.time()
        # combine_docs_chain = create_stuff_documents_chain(
        #     self.structured_llm, prompt
        # )
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # "stuff", "refine", "map_reduce" are other options
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,  # Optional: returns the source documents
        )
        print("chain created")
        response = chain.invoke(question)
        
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Time taken: {elapsed_time} seconds")
        print(response)
        return response
