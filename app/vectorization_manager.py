import os
from typing import Optional
from langchain_chroma import Chroma
from app.config import DATA_DIR, VECTOR_DIR
from app.embeddings import get_embeddings
from app.pdf_loader import load_and_split_pdf


class VectorizationManager:
    """Manages PDF vectorization and retrieval"""
    
    def __init__(self):
        self.embeddings = get_embeddings()
        os.makedirs(VECTOR_DIR, exist_ok=True)
    
    def _get_collection_name(self, pdf_name: str) -> str:
        """Convert PDF filename to collection name"""
        # Remove .pdf extension and replace spaces/special chars with underscores
        name = pdf_name.replace('.pdf', '').replace(' ', '_').replace('-', '_')
        return name.lower()
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a collection already exists"""
        try:
            # Try to load the collection
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=VECTOR_DIR,
                collection_name=collection_name
            )
            # Check if it has any documents
            return vectorstore._collection.count() > 0
        except Exception:
            return False
    
    def get_pdf_path(self, pdf_name: str) -> str:
        """Get full path to PDF file"""
        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
        return os.path.join(DATA_DIR, pdf_name)
    
    def vectorize_pdf(self, pdf_name: str, force: bool = False) -> tuple[Chroma, str]:
        """
        Vectorize a PDF and return the vectorstore
        
        Args:
            pdf_name: Name of the PDF file (with or without .pdf extension)
            force: If True, re-vectorize even if already exists
            
        Returns:
            tuple of (vectorstore, message)
        """
        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
        
        collection_name = self._get_collection_name(pdf_name)
        pdf_path = self.get_pdf_path(pdf_name)
        
        # Check if PDF file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Check if already vectorized
        if not force and self._collection_exists(collection_name):
            vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=VECTOR_DIR,
                collection_name=collection_name
            )
            return vectorstore, f"✓ Loaded existing vectorstore for '{pdf_name}'"
        
        # Vectorize the PDF
        print(f"📄 Loading and splitting PDF: {pdf_name}")
        docs = load_and_split_pdf(pdf_path)
        
        print(f"🔄 Vectorizing {len(docs)} document chunks...")
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=self.embeddings,
            persist_directory=VECTOR_DIR,
            collection_name=collection_name
        )
        
        return vectorstore, f"✓ Successfully vectorized '{pdf_name}' ({len(docs)} chunks)"
    
    def load_vectorstore(self, pdf_name: str) -> Optional[Chroma]:
        """Load an existing vectorstore for a PDF"""
        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'
        
        collection_name = self._get_collection_name(pdf_name)
        
        if not self._collection_exists(collection_name):
            return None
        
        return Chroma(
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DIR,
            collection_name=collection_name
        )
    
    def list_available_pdfs(self) -> list[str]:
        """List all PDF files in the data directory"""
        if not os.path.exists(DATA_DIR):
            return []
        
        return [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
