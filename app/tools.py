from langchain_core.tools import tool

def build_retriever_tool(retriever, pdf_name: str):
    """
    Build a retriever tool for a specific PDF
    
    Args:
        retriever: The vectorstore retriever
        pdf_name: Name of the PDF being queried
    """

    @tool
    def retriever_tool(query: str) -> str:
        """Search and retrieve information from the PDF"""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found."

        return "\n\n".join(
            f"Source {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(docs)
        )
    
    # Update the tool's name and description dynamically
    retriever_tool.name = f"search_{pdf_name.replace('.pdf', '').replace(' ', '_').replace('-', '_').lower()}"
    retriever_tool.description = f"Search and retrieve information from {pdf_name}"

    return retriever_tool
