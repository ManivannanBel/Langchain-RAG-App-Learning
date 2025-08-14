import os
import asyncio
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Import configuration
from config import RAGConfig, get_config, validate_config, print_config_summary

# Load environment variables
load_dotenv()

class WebsiteRAGChatApp:
    def __init__(self, config: RAGConfig = None):
        # Use provided config or default
        self.config = config or RAGConfig()
        
        # Validate configuration
        errors = validate_config(self.config)
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        # Initialize components with configuration
        self.llm = ChatOpenAI(**self.config.get_openai_config())
        self.embeddings = OpenAIEmbeddings(**self.config.get_embeddings_config())
        self.text_splitter = RecursiveCharacterTextSplitter(**self.config.get_text_splitter_config())
        
        self.vectorstore = None
        self.websites_data = {}
        
        # Print configuration summary if logging is enabled
        if self.config.ENABLE_LOGGING:
            print_config_summary(self.config)
        
    def fetch_website_content(self, url: str) -> Dict[str, Any]:
        """Fetch and parse website content"""
        try:
            crawler_config = self.config.get_crawler_config()
            headers = {
                'User-Agent': crawler_config['user_agent']
            }
            response = requests.get(url, headers=headers, timeout=crawler_config['timeout'])
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Limit text length if configured
            if len(text) > self.config.MAX_TEXT_LENGTH:
                text = text[:self.config.MAX_TEXT_LENGTH] + "..."
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else url
            
            return {
                "url": url,
                "title": title_text,
                "content": text,
                "status": "success"
            }
        except Exception as e:
            return {
                "url": url,
                "title": url,
                "content": "",
                "status": "error",
                "error": str(e)
            }
    
    def process_websites(self, urls: List[str]) -> Dict[str, Any]:
        """Process multiple websites and return results"""
        results = {}
        for url in urls:
            print(f"Fetching content from: {url}")
            result = self.fetch_website_content(url)
            results[url] = result
            self.websites_data[url] = result
        return results
    
    def create_documents_from_websites(self, websites_data: Dict[str, Any]) -> List[Document]:
        """Create documents from website data for vector storage"""
        documents = []
        
        for url, data in websites_data.items():
            if data["status"] == "success" and data["content"]:
                # Split content into chunks
                chunks = self.text_splitter.split_text(data["content"])
                
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "url": url,
                            "title": data["title"],
                            "chunk_id": i,
                            "source": "website"
                        }
                    )
                    documents.append(doc)
        
        return documents
    
    def setup_vectorstore(self, documents: List[Document]):
        """Setup vector store with documents"""
        if documents:
            vectorstore_config = self.config.get_vectorstore_config()
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=vectorstore_config['collection_name']
            )
            print(f"Indexed {len(documents)} document chunks in vector store")
        else:
            print("No documents to index")
    
    def search_similar_content(self, query: str, k: int = None) -> List[Document]:
        """Search for similar content in the vector store"""
        if not self.vectorstore:
            return []
        
        if k is None:
            k = self.config.SIMILARITY_SEARCH_K
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def create_rag_chain(self):
        """Create the RAG chain for answering questions"""
        # Get RAG chain configuration
        rag_config = self.config.get_rag_chain_config()
        
        # Create the prompt template
        template = rag_config['prompt_template']
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain
        # The RAG chain is a pipeline of components that process the question and return an answer.
        # The context is the relevant documents retrieved from the vector store.
        # The question is the user's question.
        # The prompt is the template for the prompt.
        # The llm is the language model used to generate the answer.
        # The StrOutputParser is used to parse the output of the llm into a string.
        rag_chain = (
            {"context": lambda x: self.search_similar_content(x["question"]), "question": lambda x: x["question"]}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def create_langgraph_workflow(self):
        """Create a LangGraph workflow for the chat application"""
        
        if not self.config.ENABLE_LANGGRAPH_WORKFLOW:
            raise ValueError("LangGraph workflow is disabled in configuration")
        
        # Define the state
        class ChatState:
            messages: List[Dict[str, Any]]
            current_question: str
            context: List[Document]
            answer: str
            websites_processed: bool
        
        # Define the nodes
        def process_websites_node(state: ChatState) -> ChatState:
            """Process websites if not already done"""
            if not state.websites_processed and hasattr(state, 'websites_to_process'):
                print("Processing websites...")
                results = self.process_websites(state.websites_to_process)
                documents = self.create_documents_from_websites(results)
                self.setup_vectorstore(documents)
                state.websites_processed = True
            return state
        
        def retrieve_context_node(state: ChatState) -> ChatState:
            """Retrieve relevant context for the question"""
            if state.current_question:
                state.context = self.search_similar_content(state.current_question)
            return state
        
        def generate_answer_node(state: ChatState) -> ChatState:
            """Generate answer using RAG chain"""
            if state.current_question and state.context:
                rag_chain = self.create_rag_chain()
                state.answer = rag_chain.invoke({"question": state.current_question})
            return state
        
        def update_messages_node(state: ChatState) -> ChatState:
            """Update the message history"""
            if state.current_question and state.answer:
                state.messages.append({
                    "role": "user",
                    "content": state.current_question
                })
                state.messages.append({
                    "role": "assistant",
                    "content": state.answer
                })
            return state
        
        # Create the workflow
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("process_websites", process_websites_node)
        workflow.add_node("retrieve_context", retrieve_context_node)
        workflow.add_node("generate_answer", generate_answer_node)
        workflow.add_node("update_messages", update_messages_node)
        
        # Define the flow
        workflow.set_entry_point("process_websites")
        workflow.add_edge("process_websites", "retrieve_context")
        workflow.add_edge("retrieve_context", "generate_answer")
        workflow.add_edge("generate_answer", "update_messages")
        workflow.add_edge("update_messages", END)
        
        if self.config.MEMORY_SAVER_ENABLED:
            return workflow.compile(checkpointer=MemorySaver())
        else:
            return workflow.compile()
    
    def chat(self, question: str, websites: Optional[List[str]] = None) -> str:
        """Main chat function"""
        if websites and not self.vectorstore:
            # Initialize with websites
            print("Initializing with websites...")
            results = self.process_websites(websites)
            documents = self.create_documents_from_websites(results)
            self.setup_vectorstore(documents)
        
        if not self.vectorstore:
            return "Please provide websites to index first."
        
        # Use the RAG chain to answer
        rag_chain = self.create_rag_chain()
        answer = rag_chain.invoke({"question": question})
        return answer
    
    def add_websites(self, urls: List[str]) -> Dict[str, Any]:
        """Add new websites to the knowledge base"""
        print(f"Adding {len(urls)} websites to the knowledge base...")
        results = self.process_websites(urls)
        
        # Create documents from new websites
        new_documents = self.create_documents_from_websites(results)
        
        if new_documents:
            if self.vectorstore:
                # Add to existing vector store
                self.vectorstore.add_documents(new_documents)
            else:
                # Create new vector store
                self.setup_vectorstore(new_documents)
        
        return results

def main():
    """Main function to run the chat application"""
    # You can specify different configurations here
    # app = WebsiteRAGChatApp(get_config("development"))
    # app = WebsiteRAGChatApp(get_config("production"))
    app = WebsiteRAGChatApp()
    
    print("🌐 Website RAG Chat Application")
    print("=" * 50)
    
    # Example websites to start with
    example_websites = [
        "https://www.moneycontrol.com/news/business/earnings/tata-motors-q1-results-net-profit-falls-30-to-rs-3-924-crore-in-line-with-estimates-13424626.html",
        "https://www.livemint.com/companies/jlr-tariff-india-demand-tata-motors-profit-uk-us-trade-pact-trump-mahindra-and-mahindra-nclat-maruti-suzuki-11754662002861.html",
        "https://www.financialexpress.com/business/industry-tata-motors-demerger-nclt-reserves-decision-company-targets-demerger-on-october-1-3940694/"
    ]
    
    print("Initializing with example websites...")
    app.add_websites(example_websites)
    
    print("\n💬 Chat with your RAG system!")
    print("Commands:")
    print("- Type 'add <url>' to add a new website")
    print("- Type 'quit' to exit")
    print("- Type your question to chat")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            elif user_input.lower().startswith('add '):
                url = user_input[4:].strip()
                if url:
                    result = app.add_websites([url])
                    if result[url]["status"] == "success":
                        print(f"✅ Successfully added {url}")
                    else:
                        print(f"❌ Failed to add {url}: {result[url].get('error', 'Unknown error')}")
                else:
                    print("Please provide a valid URL")
            
            elif user_input:
                print("\n🤖 Assistant: ", end="")
                answer = app.chat(user_input)
                print(answer)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
