import os
from typing import Union

from langchain.retrievers import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ..utilities.llm_models import get_llm_model_embedding


class VectorStoreManager:
    def __init__(self, docs_dir: str, persist_directory_dir: str, batch_size=64):
        self.embeddings = get_llm_model_embedding()
        self.vector_stores: dict[str, Union[Chroma, BM25Retriever]] = {
            "chroma": None,
            "bm25": None,
        }
        self.vs_initialized = False
        self.vector_store = None
        self.docs_dir = docs_dir
        self.persist_directory_dir = persist_directory_dir
        self.batch_size = batch_size
        self.collection_name = (
            os.getenv("OLLAM_EMB").split(":")[0].split("/")[-1].replace("-v1", "")
        )

    def initialize_vector_store(self):
        """Initialize or load the vector store"""
        chroma_vs = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory_dir,
            embedding_function=self.embeddings,
        )
        all_documents = chroma_vs.get()
        documents = [
            Document(page_content=content, id=doc_id, metadata=metadata)
            for content, doc_id, metadata in zip(
                all_documents["documents"],
                all_documents["ids"],
                all_documents["metadatas"],
            )
        ]
        bm25_vs: BM25Retriever = BM25Retriever.from_documents(documents=documents)
        self.vector_stores["chroma"] = chroma_vs
        self.vector_stores["bm25"] = bm25_vs
        self.vs_initialized = True

    def create_retriever(self, n_documents: int, bm25_portion: float = 0.4):
        self.vector_stores["bm25"].k = n_documents
        self.vector_store = EnsembleRetriever(
            retrievers=[
                self.vector_stores["bm25"],
                self.vector_stores["chroma"].as_retriever(
                    search_kwargs={"k": n_documents}
                ),
            ],
            weights=[bm25_portion, 1 - bm25_portion],
        )
        return self.vector_store

    def create_retriever(self, n_documents: int, bm25_portion: float = 0.4):
        self.vector_stores["bm25"].k = n_documents
        self.vector_store = EnsembleRetriever(
            retrievers=[
                self.vector_stores["bm25"],
                self.vector_stores["chroma"].as_retriever(
                    search_kwargs={"k": n_documents}
                ),
            ],
            weights=[bm25_portion, 1 - bm25_portion],
        )
        return self.vector_store
