import base64
import os


class PDF:
    @staticmethod
    def encode_pdf_to_base64(pdf_path):
        print(os.getcwd())
        file_path = os.getcwd() + pdf_path
        print(file_path)
        with open(file_path, "rb") as pdf_file:
            encoded_string = base64.b64encode(pdf_file.read()).decode("utf-8")
        return encoded_string
