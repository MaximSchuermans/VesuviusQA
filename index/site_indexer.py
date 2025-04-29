from langchain_community.document_loaders import RecursiveUrlLoader

import logging
import os

class ScrollSiteIndexer:
    # TODO: Lazy loading of documents
    # TODO: Add an extractor (see docs of RecursiveUrlLoader)
    # TODO: Imrpove logging with a logging_config.yaml file
    # TODO: Implement testing framework

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
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

if __name__ == "__main__":
    scraper = ScrollSiteIndexer(log_level=logging.DEBUG)
    scraper.scrape()
    for doc in scraper.documents:
        print(doc.metadata['source'])
