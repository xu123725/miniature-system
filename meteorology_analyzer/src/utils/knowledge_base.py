import os
import glob
import logging
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseManager:
    def __init__(self, 
                 kb_dir: str = "knowledge_base", 
                 index_dir: str = "data/faiss_index",
                 model_name: str = "shibing624/text2vec-base-chinese"):
        """
        Initialize the KnowledgeBaseManager.

        Args:
            kb_dir: Directory containing PDF documents.
            index_dir: Directory to save/load FAISS index.
            model_name: HuggingFace embedding model name.
        """
        self.kb_dir = kb_dir
        self.index_dir = index_dir
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store: Optional[FAISS] = None
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)
        
        # Try to load existing index
        self._load_index()

    def _load_index(self):
        """Load FAISS index from local storage if it exists."""
        try:
            if os.path.exists(self.index_dir) and os.path.exists(os.path.join(self.index_dir, "index.faiss")):
                logger.info(f"Loading existing FAISS index from {self.index_dir}...")
                self.vector_store = FAISS.load_local(
                    self.index_dir, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # Assuming local index is safe
                )
                logger.info("Index loaded successfully.")
            else:
                logger.info("No existing index found.")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            self.vector_store = None

    def build_index(self, force_rebuild: bool = False):
        """
        Build FAISS index from PDF documents in kb_dir.

        Args:
            force_rebuild: If True, rebuild index even if it already exists.
        """
        if self.vector_store is not None and not force_rebuild:
            logger.info("Index already exists. Skipping build. Use force_rebuild=True to overwrite.")
            return

        logger.info(f"Building index from documents in {self.kb_dir}...")
        
        pdf_files = glob.glob(os.path.join(self.kb_dir, "*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.kb_dir}.")
            return

        documents = []
        for file_path in pdf_files:
            try:
                logger.info(f"Loading {file_path}...")
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        if not documents:
            logger.warning("No documents loaded.")
            return

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        logger.info(f"Split into {len(texts)} chunks.")

        # Create and save index
        try:
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_store.save_local(self.index_dir)
            logger.info(f"Index saved to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to create/save index: {e}")

    def search(self, query: str, k: int = 3) -> List[str]:
        """
        Search for relevant document chunks.

        Args:
            query: The search query string.
            k: Number of results to return.

        Returns:
            List of relevant text chunks.
        """
        if self.vector_store is None:
            # Try reloading in case it was built externally or just now
            self._load_index()
            if self.vector_store is None:
                return ["Knowledge base index not found. Please build the index first."]

        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return [f"Search error: {e}"]

if __name__ == "__main__":
    # Test block
    manager = KnowledgeBaseManager()
    # Create a dummy pdf for testing if needed, or ensure kb_dir exists
    if not os.path.exists("knowledge_base"):
        os.makedirs("knowledge_base")
        print("Created knowledge_base directory. Please add PDF files there to test.")
    else:
        manager.build_index()
        results = manager.search("气象分析")
        print("Search Results:", results)
