"""Load medical documents for RAG system"""
import os
from typing import List, Dict, Any
from langchain_core.documents import Document  
from langchain_community.document_loaders import (  
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob
import tempfile
import hashlib
from datetime import datetime
import json

def load_medical_documents(data_dir: str = None) -> List[Document]:
    """
    Load medical documents from specified directory.
    
    Args:
        data_dir: Directory containing medical documents (PDF, TXT, MD, DOCX, CSV)
                  If None, will ask user for directory or use default
    
    Returns:
        List of Document objects
    """
    documents = []
    
    # If no directory specified, use default or ask
    if data_dir is None:
        data_dir = "./data"  # Default directory
        print(f"📁 Using default data directory: {data_dir}")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory '{data_dir}' not found!")
        print("💡 Please create the directory and add medical documents.")
        print("💡 Supported formats: PDF, TXT, MD, DOCX, CSV")
        return documents
    
    print(f"📂 Loading medical documents from {data_dir}...")
    
    # Helper function to load files
    def load_files(pattern, loader_class, file_type):
        files = glob.glob(os.path.join(data_dir, pattern))
        loaded_count = 0
        
        for file_path in files:
            try:
                loader = loader_class(file_path)
                file_docs = loader.load()
                
                for doc in file_docs:
                    doc.metadata["source"] = os.path.basename(file_path)
                    doc.metadata["type"] = file_type
                    doc.metadata["file_path"] = file_path
                
                documents.extend(file_docs)
                loaded_count += len(file_docs)
                print(f"   ✅ Loaded {len(file_docs)} documents from {os.path.basename(file_path)}")
                
            except Exception as e:
                print(f"   ❌ Failed to load {file_path}: {e}")
        
        return loaded_count
    
    # Load PDF files
    pdf_count = load_files("*.pdf", PyPDFLoader, "pdf")
    
    # Load text files
    txt_count = load_files("*.txt", TextLoader, "text")
    
    # Load markdown files
    md_count = load_files("*.md", UnstructuredMarkdownLoader, "markdown")
    
    # Load CSV files
    csv_count = load_files("*.csv", CSVLoader, "csv")
    
    # Load Word documents
    docx_count = 0
    docx_files = glob.glob(os.path.join(data_dir, "*.docx"))
    for docx_file in docx_files:
        try:
            loader = UnstructuredWordDocumentLoader(docx_file)
            docx_docs = loader.load()
            
            for doc in docx_docs:
                doc.metadata["source"] = os.path.basename(docx_file)
                doc.metadata["type"] = "word"
                doc.metadata["file_path"] = docx_file
            
            documents.extend(docx_docs)
            docx_count += len(docx_docs)
            print(f"   ✅ Loaded {len(docx_docs)} documents from {os.path.basename(docx_file)}")
            
        except Exception as e:
            print(f"   ❌ Failed to load {docx_file}: {e}")
    
    # Summary
    print("\n📊 DOCUMENT LOADING SUMMARY:")
    print(f"   PDF files: {pdf_count} documents")
    print(f"   Text files: {txt_count} documents")
    print(f"   Markdown files: {md_count} documents")
    print(f"   CSV files: {csv_count} documents")
    print(f"   Word documents: {docx_count} documents")
    print(f"   TOTAL: {len(documents)} documents loaded")
    
    if len(documents) == 0:
        print("\n⚠️ No documents were loaded!")
        print("💡 Please add medical documents to the data directory.")
        print("💡 Supported formats: PDF, TXT, MD, DOCX, CSV")
    
    return documents

def get_document_directory() -> str:
    """Ask user for document directory"""
    import sys
    
    default_dir = "./data"
    
    print("\n" + "="*50)
    print("📁 DOCUMENT DIRECTORY SETUP")
    print("="*50)
    print(f"Default directory: {default_dir}")
    print("\nOptions:")
    print("1. Use default directory")
    print("2. Enter custom directory path")
    print("3. Skip document loading (use existing Pinecone index)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice == "1":
        return default_dir
    elif choice == "2":
        custom_dir = input("Enter directory path: ").strip()
        if os.path.exists(custom_dir):
            return custom_dir
        else:
            print(f"❌ Directory '{custom_dir}' does not exist!")
            return default_dir
    else:
        return None  # Skip loading

def process_single_document(file_path: str, file_type: str = "pdf", user_email: str = None) -> List[Document]:
    """
    Process a single uploaded document for RAG indexing.
    
    Args:
        file_path: Path to the uploaded file
        file_type: Type of file (pdf, txt, docx, etc.)
        user_email: Email of user uploading the document
    
    Returns:
        List of Document objects with metadata
    """
    documents = []
    
    try:
        print(f"📄 Processing single document: {file_path}")
        print(f"   Type: {file_type}")
        print(f"   User: {user_email}")
        
        # Generate document ID
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:16]
        doc_id = f"doc_{file_hash}_{int(datetime.now().timestamp())}"
        
        # Load based on file type
        file_type_lower = file_type.lower().replace('application/', '').replace('.', '')
        
        if file_type_lower in ['pdf', 'application/pdf']:
            loader = PyPDFLoader(file_path)
        elif file_type_lower in ['txt', 'text', 'plain']:
            loader = TextLoader(file_path, encoding='utf-8')
        elif file_type_lower in ['md', 'markdown']:
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_type_lower == 'csv':
            loader = CSVLoader(file_path)
        elif file_type_lower in ['doc', 'docx', 'msword', 'vnd.openxmlformats-officedocument.wordprocessingml.document']:
            loader = UnstructuredWordDocumentLoader(file_path)
        else:
            # Try to auto-detect based on file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            elif file_ext == '.md':
                loader = UnstructuredMarkdownLoader(file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(file_path)
            elif file_ext in ['.doc', '.docx']:
                loader = UnstructuredWordDocumentLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type} (extension: {file_ext})")
        
        # Load the document
        file_docs = loader.load()
        
        # Add metadata to each document chunk
        for i, doc in enumerate(file_docs):
            # Preserve existing metadata
            metadata = doc.metadata.copy()
            metadata.update({
                "source": os.path.basename(file_path),
                "type": file_type_lower,
                "file_path": file_path,
                "user_email": user_email,
                "upload_time": datetime.now().isoformat(),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(file_docs),
                "file_size": os.path.getsize(file_path)
            })
            doc.metadata = metadata
        
        documents.extend(file_docs)
        print(f"✅ Successfully processed {len(file_docs)} chunks from {os.path.basename(file_path)}")
        
        return documents
        
    except Exception as e:
        print(f"❌ Error processing document {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e

def split_documents_into_chunks(documents: List[Document], 
                               chunk_size: int = 1000, 
                               chunk_overlap: int = 200) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of chunked Document objects
    """
    if not documents:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"📝 Split {len(documents)} documents into {len(chunks)} chunks")
    
    # Update chunk metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["is_chunked"] = True
        if "doc_id" in chunk.metadata:
            chunk.metadata["parent_doc_id"] = chunk.metadata["doc_id"]
    
    return chunks

def process_and_chunk_document(file_path: str, 
                              file_type: str = "pdf", 
                              user_email: str = None,
                              chunk_size: int = 1000,
                              chunk_overlap: int = 200) -> List[Document]:
    """
    Complete pipeline: Process single document and split into chunks.
    
    Args:
        file_path: Path to the uploaded file
        file_type: Type of file
        user_email: Email of user uploading
        chunk_size: Chunk size in characters
        chunk_overlap: Chunk overlap
    
    Returns:
        List of chunked Document objects
    """
    # Step 1: Load the document
    documents = process_single_document(file_path, file_type, user_email)
    
    # Step 2: Split into chunks
    if documents:
        chunks = split_documents_into_chunks(documents, chunk_size, chunk_overlap)
        return chunks
    
    return []

def save_documents_to_json(documents: List[Document], output_path: str = "documents.json"):
    """
    Save processed documents to JSON file for debugging.
    
    Args:
        documents: List of Document objects
        output_path: Path to save JSON file
    """
    try:
        data = []
        for doc in documents:
            data.append({
                "page_content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"💾 Saved {len(documents)} documents to {output_path}")
        
    except Exception as e:
        print(f"❌ Error saving documents to JSON: {e}")

def validate_document_file(file_path: str, max_size_mb: int = 50) -> Dict[str, Any]:
    """
    Validate uploaded document file.
    
    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB
    
    Returns:
        Dict with validation results
    """
    result = {
        "valid": False,
        "message": "",
        "size_mb": 0,
        "file_type": ""
    }
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            result["message"] = f"File not found: {file_path}"
            return result
        
        # Check file size
        file_size = os.path.getsize(file_path)
        file_size_mb = file_size / (1024 * 1024)
        result["size_mb"] = file_size_mb
        
        if file_size_mb > max_size_mb:
            result["message"] = f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB limit"
            return result
        
        # Check file extension
        file_ext = os.path.splitext(file_path)[1].lower()
        supported_extensions = ['.pdf', '.txt', '.md', '.csv', '.doc', '.docx']
        
        if file_ext not in supported_extensions:
            result["message"] = f"Unsupported file extension: {file_ext}. Supported: {', '.join(supported_extensions)}"
            return result
        
        # Set file type based on extension
        ext_to_type = {
            '.pdf': 'pdf',
            '.txt': 'text',
            '.md': 'markdown',
            '.csv': 'csv',
            '.doc': 'word',
            '.docx': 'word'
        }
        result["file_type"] = ext_to_type.get(file_ext, 'unknown')
        
        # Additional validation based on file type
        if file_ext == '.pdf':
            # Quick PDF validation by checking first bytes
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    result["message"] = "Invalid PDF file (missing PDF header)"
                    return result
        
        result["valid"] = True
        result["message"] = f"File validated: {os.path.basename(file_path)} ({file_size_mb:.2f}MB)"
        
    except Exception as e:
        result["message"] = f"Validation error: {str(e)}"
    
    return result

def get_document_statistics(documents: List[Document]) -> Dict[str, Any]:
    """
    Get statistics about processed documents.
    
    Args:
        documents: List of Document objects
    
    Returns:
        Dict with statistics
    """
    stats = {
        "total_documents": len(documents),
        "total_chunks": 0,
        "total_characters": 0,
        "sources": {},
        "types": {},
        "users": {}
    }
    
    if not documents:
        return stats
    
    stats["total_chunks"] = len(documents)
    
    for doc in documents:
        # Count characters
        stats["total_characters"] += len(doc.page_content)
        
        # Count by source
        source = doc.metadata.get("source", "unknown")
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        
        # Count by type
        doc_type = doc.metadata.get("type", "unknown")
        stats["types"][doc_type] = stats["types"].get(doc_type, 0) + 1
        
        # Count by user
        user = doc.metadata.get("user_email", "anonymous")
        stats["users"][user] = stats["users"].get(user, 0) + 1
    
    # Calculate average chunk size
    if stats["total_chunks"] > 0:
        stats["avg_chunk_size"] = stats["total_characters"] // stats["total_chunks"]
    
    return stats

# Example usage
if __name__ == "__main__":
    print("Testing document_loader module...")
    
    # Test loading from directory
    docs = load_medical_documents("./data")
    
    if docs:
        stats = get_document_statistics(docs)
        print("\n📊 Statistics:")
        print(f"Total documents: {stats['total_documents']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total characters: {stats['total_characters']}")
        print(f"Sources: {list(stats['sources'].keys())}")
        
        # Test chunking
        chunks = split_documents_into_chunks(docs[:2], chunk_size=500, chunk_overlap=100)
        print(f"Created {len(chunks)} chunks from first 2 documents")