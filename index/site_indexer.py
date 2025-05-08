from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma

import logging
import os
from dotenv import load_dotenv

load_dotenv()

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")


class ScrollSiteIndexer:
    def __init__(self, log_level=logging.INFO, log_file="scraper.log"):
        self.site_url = "https://scrollprize.org/"
        self.documents = []
        self.loader = RecursiveUrlLoader(self.site_url)
        # Logging
        self.logger = self._setup_logging(log_level, log_file)

    def scrape(self):
        self.logger.info("Starting scraping")
        # TODO: Implement scraping logic
        self.documents = self.loader.load()
        self.logger.info(f"Loaded {len(self.documents)} documents from {self.site_url}")
        self.logger.info("Finished scraping")

    def split_docs(self):
        self.logger.info(f"Splitting {len(self.documents)} documents")
        if len(self.documents) == 0:
            self.logger.error("No documents loaded")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        nodes = text_splitter.split_documents(self.documents)
        self.logger.info(f"Finished splitting documents into {len(nodes)} nodes")
        return nodes

    def init_index(self):
        embeddings = HuggingFaceEndpointEmbeddings(
            api_key=HUGGINGFACE_API_KEY,
            model_name="sentence-transformers/all-mpnet-base-v2",
        )
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory="../data/chroma_scrollsite_index",  # Where to save data locally, remove if not necessary
        )
        return vector_store

    def _setup_logging(self, log_level, log_file):
        """Configure and return a logger with file and console handlers."""
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(log_level)

        # Clear existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()

        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger


if __name__ == "__main__":
    index = ScrollSiteIndexer(log_level=logging.DEBUG)
    index.scrape()
    for doc in index.documents:
        print(doc.metadata["source"])
    index.split_docs()
