import shutil
from typing import Any
import evadb
import warnings
from utils import SuppressOutput, is_valid_url, loading_indicator, scrape_webpage_content
import pandas as pd
import os
import re
from gpt4all import GPT4All
import getch

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
        self.mydocs_loaded = False
        self.last_action = None

        self.separator = "="*150
        self.star_separator = "*"*150
        self.setup()

    def cleanup(self):
        if os.path.exists("evadb_data"):
            shutil.rmtree("evadb_data")

    def setup(self):
        self.cursor.query("CREATE FUNCTION SentenceFeatureExtractor IMPL 'sentence_feature_extractor.py'").df()
        self.cursor.query("CREATE TABLE mydocs (name TEXT(0), page TEXT(0), paragraph TEXT(0), data TEXT(0))").df()
        self.cursor.query("CREATE TABLE cached_queries (query TEXT(0), response TEXT(0))").df()
        self.cursor.query("CREATE TABLE documents (name TEXT(0), time_added TEXT(0))").df()

    def reset_cached_queries(self):
        self.cursor.query("DROP TABLE IF EXISTS cached_queries").df()
        self.cursor.query("CREATE TABLE cached_queries (query TEXT(0), response TEXT(0))").df()

    def get_current_time(self):
        import datetime
        import pytz

        current_time_utc = datetime.datetime.utcnow()
        eastern_us_tz = pytz.timezone('US/Eastern')
        current_time_eastern = current_time_utc.astimezone(eastern_us_tz)
        formatted_time = current_time_eastern.strftime('%Y-%m-%d %H:%M:%S %Z')

        return formatted_time

    def check_doc_loaded(self, path: str):
        df = self.cursor.query(f"SELECT * from documents WHERE name='{path}'").df()
        if not df.empty:
            return True
        else:
            return False

    def load_document(self, path: str):
        if not os.path.isfile(path):
            return False

        extension = path.split("/")[-1].split(".")[-1]
        if extension == "pdf":
            self.load_pdf(path)
            self.pdfs_loaded = True
        elif extension == "txt":
            self.load_txt(path)
            self.mydocs_loaded = True
        else:
            return False
        self.cursor.query(f"INSERT INTO documents (name, time_added) VALUES ('{path}', '{self.get_current_time()}')").df()
        return True

    def load_webpage(self, url: str):
        if not is_valid_url(url):
            return False

        paragraphs = scrape_webpage_content(url)

        for i, parapraph in enumerate(paragraphs):
            self.cursor.query(f"""
                INSERT INTO mydocs (name, page, paragraph, data) VALUES
                ('{url}', '{1}', '{i}', '{parapraph}')
            """).df()

        self.mydocs_loaded = True
        self.cursor.query(f"INSERT INTO documents (name, time_added) VALUES ('{url}', '{self.get_current_time()}')").df()

        return True

    def load_pdf(self, path: str):
        self.cursor.query(f"LOAD PDF '{path}' INTO pdfdocs").df() 

    def load_txt(self, path: str):
        paragraphs = []
        with open(path) as file:
            paragraphs = file.read().split('\n\n')

        non_empty_paragraphs  = []
        for paragraph in paragraphs:
            if paragraph != "":
                non_empty_paragraphs.append(re.sub(r'[\'";]', '', paragraph))

        for i, paragraph in enumerate(non_empty_paragraphs):
            self.cursor.query(f"""
                INSERT INTO mydocs (name, page, paragraph, data) VALUES
                ('{path}', '{1}', '{i}', '{paragraph}')
            """).df()

    def display_loaded_docs(self):
        print(self.cursor.query("""
            SELECT *
            FROM documents
        """).df())

    def setup_qdrant_index_pdf(self):
        self.cursor.query("""
            CREATE INDEX qdrant_index_pdf
            ON pdfdocs (SentenceFeatureExtractor(data))
            USING QDRANT
        """).df()

    def setup_qdrant_index_docs(self):
        self.cursor.query("""
            CREATE INDEX qdrant_index_txt
            ON mydocs (SentenceFeatureExtractor(data))
            USING QDRANT
        """).df()

    def display_results(self, results):
        print(self.separator)
        print("Here are the top matches from your documents")
        for dist, _, name, page, paragraph, data in results:
            print(self.separator)
            print("Document name:",name)
            print("Page no.:",page)
            print("Paragraph no.:",paragraph)
            print("Data:",data)

    def get_combined_results(self, result_pdf, result_txt, top_k = 5, top_k_context = 10):
        results = []
        if result_pdf is not None:
            for idx, row in result_pdf.iterrows():
                results.append((
                    row["similarity.distance"],
                    "pdf",
                    row["pdfdocs.name"],
                    row["pdfdocs.page"],
                    row["pdfdocs.paragraph"],
                    row["pdfdocs.data"],
                ))
            
        if result_txt is not None:
            for idx, row in result_txt.iterrows():
                results.append((
                    row["similarity.distance"],
                    "txt",
                    row["mydocs.name"],
                    1,
                    row["mydocs.paragraph"],
                    row["mydocs.data"],
                ))
        results.sort()
        top_k = min(top_k, len(results))
        top_k_context = min(top_k_context, len(results))
        context = ". ".join(row[5] for row in results[:top_k_context])
        return results[:top_k], context

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

    def generate_response(self, query_str: str, context: str):
        print(self.separator)
        response_str = self.check_cached(query_str)
        if response_str == "":
            print("Generating response ...")

            llm_query = f"""If the context is not relevant, please answer the question by using your own knowledge about the topic.\n                
                {context}\n
                Question : {query_str}
            """
            response_str = re.sub(r'[\'";]', '', str(self.llm.generate(llm_query)))
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
            LIMIT 10
        """

        query_txt = """
            SELECT mydocs.name, mydocs.paragraph, mydocs.data , Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            FROM mydocs
            ORDER BY Similarity(SentenceFeatureExtractor('{query_str}'), SentenceFeatureExtractor(data))
            LIMIT 10
        """
        result_pdf = None

        if self.pdfs_loaded:
            result_pdf = self.cursor.query(query_pdf).df()     

        result_txt = None

        if self.mydocs_loaded:
            result_txt = self.cursor.query(query_txt).df()

        results, context = self.get_combined_results(result_pdf, result_txt)

        self.display_results(results)
        self.generate_response(query_str, context)
        
        print("Hope you found the response helpful!")

    def prompt_message(self):
        if os.name == 'nt':
            os.system('cls')
        else:
            os.system('clear')
        print(self.star_separator)
        print("Welcome to OrgDB")
        print(self.star_separator)
        print("1. Enter 1 to load a document")
        print("2. Enter 2 to display loaded documents")
        print("3. Enter 3 to search documents")
        print("4. Enter 4 to exit")

    def __call__(self):
        while (True):
            self.prompt_message()
            mode = input("> ")
            if (mode == "1"):
                path = input("Enter the path of the document (pdf/txt/url): ")
                if self.check_doc_loaded(path):
                    print("Document already loaded!")
                elif self.load_document(path):
                    print(f"Loaded document {path}")
                    self.last_action = "load"
                    self.reset_cached_queries()
                elif self.load_webpage(url=path):
                    print(f"Loaded webpage {path}")
                    self.last_action = "load"
                    self.reset_cached_queries()
                else:
                    print("Invalid document!")
            elif (mode == "2"):
                self.display_loaded_docs()
            elif (mode == "3"):
                query_str = input("Enter your search query: ")
                self.search_query(query_str)
                self.last_action = "search_query"
            elif (mode == "4"):
                self.cleanup()
                break
            else:
                print("Invalid choice!")
            
            print("Press Enter to continue...")
            while True:
                char = getch.getch()
                if char == '\n':
                    break

def main():
    orgdb = OrgDB()
    orgdb()

if __name__ == "__main__":
    # scrape_webpage_content("https://evadb.readthedocs.io/en/stable/source/overview/getting-started.html")
    main()