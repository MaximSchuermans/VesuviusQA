from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import logging
import os
from dotenv import load_dotenv

load_dotenv()



class ScrollSiteRetriever:
    def __init__(self, log_level=logging.INFO, log_file="scraper.log", reindex=False):
        self.source_url = "https://scrollprize.org/"
        self.reindex = reindex
        self.documents = []
        self.nodes = []
        self.embedding_function = None
        self.vector_store = None
        self.retriever = None
        self.embedding_model = 'text-embedding-3-large'
        # Logging
        self.logger = self._setup_logging(log_level, log_file)

    def _scrape(self):
        self.logger.info("Starting scraping")
        loader = RecursiveUrlLoader(self.source_url)
        self.documents = loader.load()
        self.logger.info(f"Loaded {len(self.documents)} documents from {self.source_url}")
        self.logger.info("Finished scraping")

    def _load_docs(self):
        """Split loaded documents into smaller chunks."""
        self.logger.info(f"Splitting {len(self.documents)} documents")
        if len(self.documents) == 0:
            self.logger.error("No documents loaded")
            return []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )
        self.splits = text_splitter.split_documents(self.documents)
        self.logger.info(f"Finished splitting documents into {len(self.splits)} nodes")
        return self.splits

    def init(self):
        """Initialize embedding function and vector store, optionally reindexing."""
        self.logger.info("Initialising embedding function ...")
        self.embedding_function = OpenAIEmbeddings(model=self.embedding_model)

        index_path = "../data/chroma_scrollsite_index"

        if not self.reindex and os.path.exists(index_path):
            self.logger.info("Loading existing vector store ...")
            self.vector_store = Chroma(
                collection_name="example_collection",
                embedding_function=self.embedding_function,
                persist_directory=index_path,
            )
            self.logger.info("Loaded existing vector store")
            return self.vector_store

        self._scrape()
        self._load_docs()
        self.logger.info("Initialising vector store ...")
        self.vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=self.embedding_function,
            persist_directory=index_path,
        )
        self.logger.info(f"Saving {len(self.splits)} splits to vector store {self.vector_store}")
        self.vector_store.add_documents(documents=self.splits)
        self.logger.info("Initialised scroll site vector store")
        self.logger.info("Initialising retriever from vector store ...")
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k":3})
        return self.vector_store

    def retrieve(self, query_list):
        self.logger.info(f"Retrieving top 3 docs for {len(query_list)} queries ...")
        return self.retriever.batch(query_list)

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
    retriever = ScrollSiteRetriever(reindex=True, log_level=logging.DEBUG)
    retriever.init()
    for doc in retriever.documents:
        print(doc.metadata["source"])
    print(retriever.retriever)
    retriever.retrieve(["What is the vesuvius competition"])
