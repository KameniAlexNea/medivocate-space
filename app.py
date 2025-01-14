import os
from typing import List

import gradio as gr

from src.rag_pipeline.rag_system import RAGSystem

# Set environment variable to optimize tokenization performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ChatInterface:
    """Interface for interacting with the RAG system via Gradio's chat component."""

    def __init__(self, rag_system: RAGSystem):
        self.rag_system = rag_system

    def respond(self, message: str, history: List[dict]):
        """
        Processes a user message and returns responses incrementally using the RAG system.

        Args:
            message (str): User's input message.
            history (List[dict]): Chat history as a list of role-content dictionaries.

        Yields:
            str: Incremental response generated by the RAG system.
        """
        # Convert history to (role, content) tuples and limit to the last 10 turns
        processed_history = [(turn["role"], turn["content"]) for turn in history][-10:]
        result = ""

        # Generate response incrementally
        for text in self.rag_system.query(message, processed_history):
            result += text
            yield result

    def create_interface(self) -> gr.ChatInterface:
        """
        Creates the Gradio chat interface for Medivocate.

        Returns:
            gr.ChatInterface: Configured Gradio chat interface.
        """
        description = (
            "Medivocate is an application that offers clear and structured information "
            "about African history and traditional medicine. The knowledge is exclusively "
            "based on historical documentaries about the African continent.\n\n"
            "🌟 **Code Repository**: [Medivocate GitHub](https://github.com/KameniAlexNea/medivocate)"
        )
        return gr.ChatInterface(
            fn=self.respond,
            type="messages",
            title="Medivocate",
            description=description,
        )

    def launch(self, share: bool = False):
        """
        Launches the Gradio interface.

        Args:
            share (bool): Whether to generate a public sharing link. Defaults to False.
        """
        interface = self.create_interface()
        interface.launch(share=share)


# Entry point
if __name__ == "__main__":
    # Initialize the RAG system with specified parameters
    top_k_documents = 12
    rag_system = RAGSystem(top_k_documents=top_k_documents)
    rag_system.initialize_vector_store()

    # Create and launch the chat interface
    chat_interface = ChatInterface(rag_system)
    chat_interface.launch(share=False)
