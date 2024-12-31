import os
import logging
from typing import Optional, Dict
from markitdown import MarkItDown
import base64
import requests
from PIL import Image
import json

class ContentManager:
    def __init__(self):
        self.md = MarkItDown()

    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a single file using MarkItDown or OCR for images.

        :param file_path: Path to the file.
        :return: Extracted text as a string, or None if an error occurs.
        """
        _, extension = os.path.splitext(file_path)
        if extension.lower() in [".jpg", ".png"]:
            return self.perform_ocr(file_path)
        try:
            logging.info(f"Processing file: {file_path}")
            result = self.md.convert(file_path)
            return result.text_content
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            return None

    def process_directory(self, directory_path: str, file_extensions: Optional[list[str]] = None) -> Dict[str, str]:
        """
        Recursively process all files in a directory to extract text.

        :param directory_path: Path to the directory.
        :param file_extensions: List of file extensions to process (e.g., ['.md', '.txt']).
        :return: Dictionary mapping file paths to extracted text.
        """
        extracted_texts = {}
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                    continue
                file_path = os.path.join(root, file)
                text = self.extract_text(file_path)
                if text:
                    extracted_texts[file_path] = text
        return extracted_texts


    def perform_ocr(self, image_path: str) -> Optional[str]:
        """Perform OCR on the given image using Llama 3.2-Vision."""
        try:
            base64_image = self.encode_image_to_base64(image_path)
            ollama_host = os.getenv("OLLAMA_CHAT_API_URL", "http://localhost:11434/chat")
            response = requests.post(
                f"{ollama_host}",
                json={
                    "model": "llama3.2-vision",
                    "messages": [
                        {
                            "role": "user",
                            "content": "Mit látsz a képen? Ha szöveges akkor add vissza pontosan a teljes szöveget",
                            "images": [base64_image],
                        },
                    ],
                },
                timeout=60
            )

            # Naplózd a teljes választ
            #logging.info(f"OCR response: {response.text}")

            if response.status_code != 200:
                logging.error(f"OCR Error: {response.status_code} {response.text}")
                return None

            # Iteráljunk az egyes JSON-objektumokon
            text_content = ""
            for line in response.text.splitlines():
                try:
                    json_obj = json.loads(line)
                    content = json_obj.get("message", {}).get("content", "")
                    text_content += content
                except json.JSONDecodeError as e:
                    logging.error(f"JSON decoding error: {e} | Raw line: {line}")
                    continue

            # Térj vissza a teljes OCR szöveggel
            return text_content.strip() if text_content else None

        except requests.exceptions.ConnectionError:
            logging.error("Failed to connect to Ollama Docker container. Is it running?")
            return None
        except Exception as e:
            logging.error(f"Error performing OCR on {image_path}: {e}")
            return None

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """Convert an image file to a base64 encoded string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

if __name__ == "__main__":
    cm = ContentManager()

    # Példa kép OCR feldolgozása
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    result = cm.perform_ocr(image_path)
    if result:
        print("OCR Recognition Result:")
        print(result)

    # Példa könyvtár feldolgozása
    directory_to_process = "path/to/your/directory"
    allowed_extensions = ['.xlsx', '.docx', '.pptx', '.jpg', '.png']  # Add extensions based on your needs
    texts = cm.process_directory(directory_to_process, allowed_extensions)
    for file_path, text in texts.items():
        print(f"Extracted text from {file_path}:")
        print(text)
        print("-" * 40)

