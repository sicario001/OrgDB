import os
import re
import sys
from tqdm import tqdm
from functools import wraps
import requests
from bs4 import BeautifulSoup

RESET = "\033[0m"

BLINK = "\033[5m"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

TMP_PATH = "tmp"

class SuppressOutput:
    def __enter__(self):
        # Save a copy of the current file descriptors for stdout and stderr
        self.stdout_fd = os.dup(1)
        self.stderr_fd = os.dup(2)

        # Open a file to /dev/null
        self.devnull_fd = os.open(os.devnull, os.O_WRONLY)

        # Replace stdout and stderr with /dev/null
        os.dup2(self.devnull_fd, 1)
        os.dup2(self.devnull_fd, 2)

        # Writes to sys.stdout and sys.stderr should still work
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = os.fdopen(self.stdout_fd, "w")
        sys.stderr = os.fdopen(self.stderr_fd, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore stdout and stderr to their original state
        os.dup2(self.stdout_fd, 1)
        os.dup2(self.stderr_fd, 2)

        # Close the saved copies of the original stdout and stderr file descriptors
        os.close(self.stdout_fd)
        os.close(self.stderr_fd)

        # Close the file descriptor for /dev/null
        os.close(self.devnull_fd)

        # Restore sys.stdout and sys.stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

def is_valid_url(url: str):
    try:
        response = requests.get(url)
    except Exception:
        return False
    return response.status_code == 200

def scrape_webpage_content(url: str):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        page_content = soup.get_text()
        paragraphs = page_content.split('\n\n')
        
        non_empty_paragraphs  = []
        for paragraph in paragraphs:
            if paragraph != "":
                non_empty_paragraphs.append(re.sub(r'[\'";]', '', paragraph))

        return non_empty_paragraphs
    else:
        return None