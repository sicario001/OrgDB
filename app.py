import shutil
from typing import Any
import evadb
import warnings
from utils import SuppressOutput, loading_indicator
import pandas as pd
import os
import re
from gpt4all import GPT4All

class OrgDB:
    @loading_indicator(start_message="Initializing OrgDB...", end_message="Initialization complete")
    def __init__(self):
        self.cleanup()

        with SuppressOutput():
            self.llm = GPT4All("ggml-model-gpt4all-falcon-q4_0.bin")

        pd.set_option('display.max_colwidth', None)
        self.cursor = evadb.connect().cursor()
        warnings.filterwarnings("ignore")

        self.pdfs_loaded = False
        self.txts_loaded = False
        self.last_action = None

        self.separator = "*"*150
        self.setup()

    def cleanup(self):
        """Removes any temporary file / directory created by EvaDB."""
        if os.path.exists("evadb_data"):
            shutil.rmtree("evadb_data")

    def setup(self):
        self.cursor.query("CREATE FUNCTION SentenceFeatureExtractor IMPL 'sentence_feature_extractor.py'").df()
        self.cursor.query("CREATE TABLE txtdocs (name TEXT(0), page TEXT(0), paragraph TEXT(0), data TEXT(0))").df()
        self.cursor.query("CREATE TABLE cached_queries (query TEXT(0), response TEXT(0))").df()

    def reset_cached_queries(self):
        self.cursor.query("DROP TABLE IF EXISTS cached_queries").df()
        self.cursor.query("CREATE TABLE cached_queries (query TEXT(0), response TEXT(0))").df()

    def load_document(self, path: str):
        if not os.path.isfile(path):
            print("File doesn't exist!")
            return False
        extension = path.split("/")[-1].split(".")[-1]
        if extension == "pdf":
            self.load_pdf(path)
            self.pdfs_loaded = True
        elif extension == "txt":
            self.load_txt(path)
            self.txts_loaded = True
        else:
            print("Invalid file type!")
            return False
        return True

    def load_pdf(self, path: str):
        self.cursor.query(f"LOAD PDF '{path}' INTO pdfdocs").df()
        self.setup_qdrant_index_pdf()  

    def load_txt(self, path: str):
        lines = []
        with open(path) as file:
            lines = file.readlines()
        for i, line in enumerate(lines):
            line = re.sub(r'[^0-9a-zA-Z ]+', '', line).strip()
            if line == "":
                continue
            self.cursor.query(f"""
                INSERT INTO txtdocs (name, page, paragraph, data) VALUES
                ('{path}', '{1}', '{i}', '{line}')
            """).df()
        self.setup_qdrant_index_txt()

    def setup_qdrant_index_pdf(self):
        self.cursor.query("""
            CREATE INDEX qdrant_index_pdf
            ON pdfdocs (SentenceFeatureExtractor(data))
            USING QDRANT
        """).df()

    def setup_qdrant_index_txt(self):
        self.cursor.query("""
            CREATE INDEX qdrant_index_txt
            ON txtdocs (SentenceFeatureExtractor(data))
            USING QDRANT
        """).df()

    def display_pdf_results(self, result_pdf):
        print(self.separator)
        print("Here are the top matches from your PDFs")
        for idx, row in result_pdf.iterrows():
            print(self.separator)
            print("PDF name:",row["pdfdocs.name"])
            print("Page no.:",row["pdfdocs.page"])
            print("Paragraph no.:",row["pdfdocs.paragraph"])
            print("Data:",row["pdfdocs.data"])

    def display_txt_results(self, result_txt):
        print(self.separator)
        print("Here are the top matches from your txts")
        for idx, row in result_txt.iterrows():
            print(self.separator)
            print("TXT name:",row["txtdocs.name"])
            print("Paragraph no.:",row["txtdocs.paragraph"])
            print("Data:",row["txtdocs.data"])

    def check_cached(self, query_str: str):
        query = f"""
            SELECT query, response, Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(query))
            FROM cached_queries
            WHERE Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(query)) < 0.2
            ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(query))
            LIMIT 1
        """
        df = self.cursor.query(query).df()
        if not df.empty:
            return df["cached_queries.response"].str.cat()
        else:
            return ""

    def cache_query(self, query_str: str, response_str: str):
        query = f"""
            INSERT INTO
            cached_queries (query, response)
            VALUES
            ('{query_str}', '{response_str}');
        """
        self.cursor.query(query).df()

    def generate_response(self, query_str: str, context_pdf: str, context_txt: str):
        print(self.separator)
        response_str = self.check_cached(query_str)
        if response_str == "":
            print("Generating response ...")

            llm_query = f"""Answer the query based on the provided context.
                context : {context_pdf}. {context_txt}
                query : {query_str}
            """
            response_str = str(self.llm.generate(llm_query))
        else:
            print("Using cached response")

        self.cache_query(query_str, response_str)

        print(self.separator)
        print(response_str)
        print(self.separator)

    def search_query(self, query_str: str):
        query_pdf = f"""
            SELECT pdfdocs.name, pdfdocs.page, pdfdocs.paragraph, pdfdocs.data , Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            FROM pdfdocs
            ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            LIMIT 5
        """

        query_txt = """
            SELECT txtdocs.name, txtdocs.paragraph, txtdocs.data , Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            FROM txtdocs
            ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            LIMIT 5
        """
        if self.pdfs_loaded:
            result_pdf = self.cursor.query(query_pdf).df()
            self.display_pdf_results(result_pdf)
            context_pdf = result_pdf["pdfdocs.data"].str.cat(sep = ". ")
        else:
            context_pdf = ""

        if self.txts_loaded:
            result_txt = self.cursor.query(query_txt).df()
            self.display_txt_results(result_txt)
            context_txt = result_txt["txtdocs.data"].str.cat(sep = ". ")
        else:
            context_txt = ""

        self.generate_response(query_str, context_pdf, context_txt)
        
        print("Hope you found the response helpful!")

    def prompt_message(self):
        print(self.separator)
        print("1. Enter 1 to upload your organization's document")
        print("2. Enter 2 to search your documents")
        print("3. Exit")

    def __call__(self):
        print(self.separator)
        print("Welcome to OrgDB")
        while (True):
            self.prompt_message()
            mode = input("Mode: ")
            if (mode == "1"):
                path = input("Enter the path of the document (pdf/txt): ")
                if self.load_document(path):
                    print(f"Loaded {path}")
                    self.last_action = "load_document"
                    self.reset_cached_queries()
            elif (mode == "2"):
                query_str = input("Enter your search query: ")
                self.search_query(query_str)
                self.last_action = "search_query"
            elif (mode == "3"):
                self.cleanup()
                break
            else:
                print("Invalid choice!")

def main():
    orgdb = OrgDB()
    orgdb()

if __name__ == "__main__":
    main()