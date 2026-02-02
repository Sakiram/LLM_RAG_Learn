from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import MODEL, OPENAI_API_KEY
from app.vectorization_manager import VectorizationManager
from app.tools import build_retriever_tool
from app.graph import build_graph


def run():
    """Interactive PDF Q&A with vectorization"""
    
    manager = VectorizationManager()
    current_pdf = None
    current_vectorstore = None
    graph = None
    
    print("\n" + "="*60)
    print("📚 PDF Question & Answer System")
    print("="*60)
    
    # Show available PDFs
    available_pdfs = manager.list_available_pdfs()
    if available_pdfs:
        print(f"\n📁 Available PDFs in data folder:")
        for pdf in available_pdfs:
            print(f"   • {pdf}")
    else:
        print("\n⚠️  No PDFs found in data folder")
    
    print("\n💡 Commands:")
    print("   • vectorize <filename> - Vectorize a PDF")
    print("   • list - Show available PDFs")
    print("   • current - Show currently loaded PDF")
    print("   • exit/quit - Exit the program")
    print("   • Or just ask a question about the loaded PDF")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            # Exit command
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Goodbye!")
                break
            
            # List PDFs command
            if user_input.lower() == "list":
                available_pdfs = manager.list_available_pdfs()
                if available_pdfs:
                    print(f"\n📁 Available PDFs:")
                    for pdf in available_pdfs:
                        print(f"   • {pdf}")
                else:
                    print("\n⚠️  No PDFs found in data folder")
                continue
            
            # Show current PDF
            if user_input.lower() == "current":
                if current_pdf:
                    print(f"\n📄 Currently loaded: {current_pdf}")
                else:
                    print("\n⚠️  No PDF currently loaded. Use 'vectorize <filename>' first.")
                continue
            
            # Vectorize command
            if user_input.lower().startswith("vectorize "):
                pdf_name = user_input[10:].strip()
                
                if not pdf_name:
                    print("\n⚠️  Please provide a PDF filename. Example: vectorize myfile.pdf")
                    continue
                
                try:
                    vectorstore, message = manager.vectorize_pdf(pdf_name)
                    print(f"\n{message}")
                    
                    # Update current PDF and rebuild graph
                    current_pdf = pdf_name if not pdf_name.endswith('.pdf') else pdf_name
                    if not current_pdf.endswith('.pdf'):
                        current_pdf += '.pdf'
                    current_vectorstore = vectorstore
                    
                    # Build retriever and graph
                    retriever = current_vectorstore.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 3}
                    )
                    retriever_tool = build_retriever_tool(retriever, current_pdf)
                    
                    llm = ChatOpenAI(
                        model=MODEL,
                        api_key=OPENAI_API_KEY
                    ).bind_tools([retriever_tool])
                    
                    graph = build_graph(llm, [retriever_tool])
                    
                    print(f"✓ Ready to answer questions about '{current_pdf}'")
                    
                except FileNotFoundError as e:
                    print(f"\n❌ Error: {e}")
                    print(f"   Make sure the PDF is in the 'data' folder")
                except Exception as e:
                    print(f"\n❌ Error vectorizing PDF: {e}")
                
                continue
            
            # Question answering
            if not current_pdf or not graph:
                print("\n⚠️  Please vectorize a PDF first using: vectorize <filename>")
                continue
            
            # Process the question
            print(f"\n🤔 Searching '{current_pdf}'...")
            result = graph.invoke({
                "messages": [HumanMessage(content=user_input)]
            })
            
            print("\n" + "="*60)
            print("💬 ANSWER:")
            print("="*60)
            print(result["messages"][-1].content)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")