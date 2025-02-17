import configparser
import os
from dotenv import load_dotenv  # type: ignore

import utils.anthropic_citations as anthropic_citations # type: ignore
import utils.openai_langchain_PDFchat as openai_langchain_PDFchat # type: ignore
from utils.pdf import PDF
import time

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC = anthropic_citations.Anthropic_Citations()
OPENAI = openai_langchain_PDFchat.OpenAI_LangChain()

PDF_PATH = "/Boston - Mobile App Development RFP.pdf"
Q1 = "What are the deliverables?"
Q2 = "Give me a list of key dates."
Q3 = "Is there anything that is not straight forward or normal in the context of RFPs?"


def main():
    pdf_data = PDF.encode_pdf_to_base64(PDF_PATH)

    # ANTHROPIC.get_citations(pdf_data, Q1)
    # time.sleep(20) # To ensure we don't rate limit
    # ANTHROPIC.get_citations(pdf_data, Q2)
    # time.sleep(20)
    # ANTHROPIC.get_citations(pdf_data, Q3)
    
    # OPENAI.get_citations(PDF_PATH, Q1)
    # time.sleep(20) # To ensure we don't rate limit
    # OPENAI.get_citations(PDF_PATH, Q2)
    # time.sleep(20) # To ensure we don't rate limit
    # OPENAI.get_citations(PDF_PATH, Q3)

    


if __name__ == "__main__":
    main()
