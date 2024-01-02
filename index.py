import os
from pathlib import Path

from langchain.text_splitter import CharacterTextSplitter

def chunk_section(section, chunk_size, chunk_overlap):
    text_spliter = CharacterTextSplitter(
        separators = '\n',
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_funtion = len
    )
    chunk = text_spliter.create_document(section)
    return chunk

