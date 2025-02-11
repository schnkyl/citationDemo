import base64
import time
import anthropic  # type: ignore


class Anthropic_Citations:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def get_citations(self, pdf_data, question):
        start_time = time.time()
        try:
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": pdf_data,
                                },
                                "title": "Boston - Mobile App Development RFP.pdf",
                                "context": "This is a trustworthy document.",
                                "citations": {"enabled": True},
                                "cache_control": {"type": "ephemeral"},
                            },
                            {
                                "type": "text",
                                "text": question,
                            },
                        ],
                    }
                ],
            )
        except Exception as e:
            print(f"An error occurred: {e}")
            response = None
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        print(f"Time taken: {elapsed_time} seconds")
        print(response)
        print("\n\n\n")
        return response
