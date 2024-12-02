import pathlib
import docx2txt
import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer



def extract_text_from_doc(filename) :
        '''
        Extract txt from the incomming document based on the file type.
        It is too small to be a ray remote, so part of the pre-process
        :param file_content: bytearray: Binary content of the received file.
        :param type: str: The type of the file
        '''
        def clean_text(txt):
            # Merge hyphenated words
            text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt.strip())
            # Fix newlines in the middle of sentences
            text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
            # Remove multiple newlines
            text = re.sub(r"\n\s*\n", "\n\n", text)
            # Remove tabs
            text  = text.replace("\t", " ")
            return text
            
            
        def _extract_from_txt(doc):
            if doc:
                doc = doc.decode("utf-8")
                doc = clean_text(doc)
                return doc
            return None

        def _extract_from_docx(docx):
            doc = BytesIO(docx)
            doc = docx2txt.process(doc)
            if doc:
                doc = clean_text(doc)
                return doc
            return None

        def _extract_from_pdf(pdf):
            doc = BytesIO(pdf)
            all_pages_text = []

            for page_layout in extract_pages(doc):
                page_text = ""
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        page_text += clean_text(element.get_text()) + " "
                all_pages_text.append(page_text)

            doc = " ".join(all_pages_text)
            if doc:
                return doc
            return None

        ext = filename.name.split('.')[-1]
        with open(filename, mode='rb') as file:
            file_content = file.read()
        if ext == 'pdf' :
            doc = _extract_from_pdf(file_content)
        elif ext == 'docx' :
            doc = _extract_from_docx(file_content)
        elif ext == 'txt' :
            doc= _extract_from_txt(file_content)
        else:
            return None
        return doc

