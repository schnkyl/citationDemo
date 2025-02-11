import configparser
import os
from dotenv import load_dotenv  # type: ignore

import utils.anthropic_citations as anthropic_citations # type: ignore
from src.utils.pdf import PDF

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ANTHROPIC = anthropic_citations.Anthropic_Citations()

PDF_PATH = "/Boston - Mobile App Development RFP.pdf"
Q1 = "What are the deliverables?"
Q2 = "Give me a list of key dates."
Q3 = "Is there anything that is not straight forward or normal in the context of RFPs?"


def main():
    pdf_data = PDF.encode_pdf_to_base64(PDF_PATH)

    ### If you run all three you may get a rate limit, we will have to up it rate_limit_error
    # ANTHROPIC.get_citations(pdf_data, Q1)
    # ANTHROPIC.get_citations(pdf_data, Q2)
    # ANTHROPIC.get_citations(pdf_data, Q3)


if __name__ == "__main__":
    main()
