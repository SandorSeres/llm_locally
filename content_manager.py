import os
import logging
from typing import Optional, Dict
from markitdown import MarkItDown

class ContentManager:
    def __init__(self):
        self.md = MarkItDown()

    def extract_text(self, file_path: str) -> Optional[str]:
        """
        Extract text from a single file using MarkItDown.

        :param file_path: Path to the file.
        :return: Extracted text as a string, or None if an error occurs.
        """
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

if __name__ == "__main__":
    # Example usage
    directory_to_process = "path/to/your/directory"
    allowed_extensions = ['.xlsx', '.docx', '.pptx']  # Add extensions based on your needs
    cm = ContentManager()
    texts = cm.process_directory(directory_to_process, allowed_extensions)
    for file_path, text in texts.items():
        print(f"Extracted text from {file_path}:")
        print(text)
        print("-" * 40)

