import evadb
import warnings
import pandas as pd
import os
import re
from gpt4all import GPT4All

llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")
pd.set_option('display.max_colwidth', None)
cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")
pdfs_loaded = False

separator = "*"*150

def setup():
    cursor.query("DROP TABLE IF EXISTS pdfdocs").df()
    cursor.query("DROP TABLE IF EXISTS txtdocs").df()
    cursor.query("DROP FUNCTION IF EXISTS SentenceFeatureExtractor").df()
    cursor.query("CREATE FUNCTION SentenceFeatureExtractor IMPL 'sentence_feature_extractor.py'").df()
    cursor.query("CREATE TABLE txtdocs (name TEXT(0), page TEXT(0), paragraph TEXT(0), data TEXT(0))").df()

def load_document(path: str):
    if not os.path.isfile(path):
        print("File doesn't exist!")
        return
    extension = path.split("/")[-1].split(".")[-1]
    if extension == "pdf":
        load_pdf(path)
    elif extension == "txt":
        load_txt(path)
    else:
        print("Invalid file type!")

def load_pdf(path: str):
    global pdfs_loaded
    cursor.query(f"LOAD PDF '{path}' INTO pdfdocs").df()
    setup_qdrant_index_pdf()
    pdfs_loaded = True
    print(f"Loaded {path}")

def load_txt(path: str):
    lines = []
    with open(path) as file:
        lines = file.readlines()
    for i, line in enumerate(lines):
        line = re.sub(r'[^0-9a-zA-Z ]+', '', line).strip()
        if line == "":
            continue
        cursor.query(f"""
            INSERT INTO txtdocs (name, page, paragraph, data) VALUES
            ('{path}', '{1}', '{i}', '{line}')
        """).df()
    setup_qdrant_index_txt()
    print(f"Loaded {path}")

def setup_qdrant_index_pdf():
    cursor.query("""
        CREATE INDEX qdrant_index_pdf
        ON pdfdocs (SentenceFeatureExtractor(data))
        USING QDRANT
    """).df()

def setup_qdrant_index_txt():
    cursor.query("""
        CREATE INDEX qdrant_index_txt
        ON txtdocs (SentenceFeatureExtractor(data))
        USING QDRANT
    """).df()

def display_pdf_results(result_pdf):
    print(separator)
    print("Here are the top matches from your PDFs")
    for idx, row in result_pdf.iterrows():
        print(separator)
        print("PDF name:",row["pdfdocs.name"])
        print("Page no.:",row["pdfdocs.page"])
        print("Paragraph no.:",row["pdfdocs.paragraph"])
        print("Data:",row["pdfdocs.data"])

def display_txt_results(result_txt):
    print(separator)
    print("Here are the top matches from your txts")
    for idx, row in result_txt.iterrows():
        print(separator)
        print("TXT name:",row["txtdocs.name"])
        print("Paragraph no.:",row["txtdocs.paragraph"])
        print("Data:",row["txtdocs.data"])

def search_query(query_str: str):
    global pdfs_loaded
    query_pdf = f"""
        SELECT *
        FROM pdfdocs
        ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
        LIMIT 5
    """

    query_txt = """
        SELECT *
        FROM txtdocs
        ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
        LIMIT 5
    """
    if pdfs_loaded:
        result_pdf = cursor.query(query_pdf).df()
        display_pdf_results(result_pdf)
        context_pdf = result_pdf["pdfdocs.data"].str.cat(sep = ". ")
    else:
        context_pdf = ""
    
    result_txt = cursor.query(query_txt).df()
    display_txt_results(result_txt)
    context_txt = result_txt["txtdocs.data"].str.cat(sep = ". ")

    llm_query = f"""Answer the question based on the provided context. If the context is not relevant, please answer the question by using your own knowledge about the topic.
        context : {context_pdf}. {context_txt}
        question : {query_str}
    """
    full_response = llm.generate(llm_query)
    print(separator)
    print(full_response)
    print(separator)
    print("Hope you found the response helpful!")

def prompt_message():
    print(separator)
    print("1. Enter 1 to upload your organization's document")
    print("2. Enter 2 to search your documents")
    print("3. Exit")

def OrgDB():
    setup()
    print(separator)
    print("Welcome to OrgDB")
    while (True):
        prompt_message()
        mode = input("Mode: ")
        if (mode == "1"):
            path = input("Enter the path of the document (pdf/txt): ")
            load_document(path)
        elif (mode == "2"):
            query_str = input("Enter your search query: ")
            search_query(query_str)
        elif (mode == "3"):
            break
        else:
            print("Invalid choice!")
def main():
    OrgDB()

if __name__ == "__main__":
    main()