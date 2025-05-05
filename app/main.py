# === Local Python Code (Adapted from Colab - V3 Final Enhanced Browser) ===

# --- Necessary Imports ---
import os
import logging
import re # Regex library
import datetime
import uuid
import random  # For random provider selection
from typing import List, Dict, Any, Optional, TypedDict, Union
import traceback
# Environment Variable Loading (Add this!)
from dotenv import load_dotenv

# FastAPI Imports (Add this!)
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel # For request/response models

# LangChain / LangGraph Imports
# Replacing Ollama with remote LLM providers
# from langchain_ollama import OllamaLLM as Ollama
# from langchain_ollama import OllamaEmbeddings
from langchain_groq import ChatGroq
from langchain_together import ChatTogether
from langchain.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_core.language_models import BaseChatModel

# Database Imports
from pymongo import MongoClient
from bson import ObjectId

# Opik/Comet Integration
import opik
from opik.integrations.langchain import OpikTracer

# Uvicorn Import (Add this!)
import uvicorn

# --- Import policies module
import app.policies

# --- Load Environment Variables ---
load_dotenv() # Load variables from .env file in the project root

# --- Configuration from Environment Variables ---
MONGO_URI = os.getenv('MONGO_URI')
COMET_API_KEY = os.getenv('COMET_API_KEY')
OPIK_PROJECT_NAME = os.getenv('OPIK_PROJECT_NAME', "cuny-suny-assistant-chatbot-chat") # Default if not set

# LLM API Keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')

# Validate required environment variables
if not MONGO_URI:
    print("CRITICAL WARNING: MONGO_URI environment variable not set. Database functionality will fail.")
    # Depending on requirements, you might want to raise an error here:
    # raise ValueError("MONGO_URI environment variable is required.")
if not COMET_API_KEY:
    print("WARNING: COMET_API_KEY environment variable not set. Opik tracing will be disabled.")
if not OPIK_PROJECT_NAME:
     print("WARNING: OPIK_PROJECT_NAME environment variable not set. Using default project name.")
     
# Validate LLM API keys
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not set. Groq provider will be disabled.")
if not CEREBRAS_API_KEY:
    print("WARNING: CEREBRAS_API_KEY not set. Cerebras provider will be disabled.")
if not TOGETHER_API_KEY:
    print("WARNING: TOGETHER_API_KEY not set. Together AI provider will be disabled.")

# --- Track LLM usage (basic implementation) ---
llm_usage_counter = {
    "groq": 0,
    "cerebras": 0,
    "together": 0
}
# Reset counts at midnight
last_reset_day = datetime.datetime.now().day

# --- LLM Provider Class for Load Balancing ---
class LLMProvider:
    def __init__(self):
        self.llm_providers = []
        self._initialize_providers()
        self.current_index = 0
        
    def _initialize_providers(self):
        """Initialize all available LLM providers based on API keys"""
        # Add Groq if API key is available
        if GROQ_API_KEY:
            try:
                groq_llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.7,
                    max_tokens=1024,
                    max_retries=2
                )
                self.llm_providers.append(("groq", groq_llm))
                print("Groq LLM initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Groq LLM")
        
        # Add Cerebras if API key is available
        if CEREBRAS_API_KEY:
            try:
                cerebras_llm = ChatOpenAI(
                    model_name="llama-4-scout-17b-16e-instruct",
                    openai_api_base="https://api.cerebras.ai/v1",
                    openai_api_key=CEREBRAS_API_KEY,
                    temperature=0.7,
                    max_tokens=1024
                )
                self.llm_providers.append(("cerebras", cerebras_llm))
                print("Cerebras LLM initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Cerebras LLM")
        
        # Add Together AI if API key is available
        if TOGETHER_API_KEY:
            try:
                together_llm = ChatTogether(
                    model="meta-llama/Llama-3-8b-chat-hf",
                    temperature=0.7,
                    max_tokens=1024
                )
                self.llm_providers.append(("together", together_llm))
                print("Together AI LLM initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Together AI LLM")
        
        if not self.llm_providers:
            raise ValueError("No LLM providers could be initialized. Ensure at least one API key is valid.")
    
    def get_llm(self) -> tuple[str, BaseChatModel]:
        """Return the next LLM provider in rotation"""
        global llm_usage_counter, last_reset_day
        
        # Reset counts if it's a new day
        current_day = datetime.datetime.now().day
        if current_day != last_reset_day:
            llm_usage_counter = {"groq": 0, "cerebras": 0, "together": 0}
            last_reset_day = current_day
        
        # Get next provider
        provider_name, provider_llm = self.llm_providers[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.llm_providers)
        
        # Update usage counter
        llm_usage_counter[provider_name] += 1
        
        return provider_name, provider_llm
    
    def invoke(self, prompt, state=None, **kwargs):
        """Invoke an LLM with fallback if the primary fails"""
        # Try providers in sequence until one succeeds
        errors = []
        
        # Track if state has metadata field
        has_metadata = state is not None and isinstance(state, dict) and "system_metadata" in state
        if has_metadata:
            print(f"State has metadata before LLM invocation: {state['system_metadata']}")
        else:
            print("State does not have metadata field or is None")
        
        # Try up to 3 different providers before giving up
        for _ in range(min(3, len(self.llm_providers))):
            provider_name, provider_llm = self.get_llm()
            try:
                print(f"Using {provider_name} LLM provider")
                response = provider_llm.invoke(prompt, **kwargs)
                
                # Update system metadata in state if provided
                if has_metadata:
                    # Determine model name based on provider
                    if provider_name == "groq":
                        model_name = "llama-3.1-8b-instant"
                    elif provider_name == "cerebras":
                        model_name = "llama-4-scout-17b-16e-instruct"
                    elif provider_name == "together":
                        model_name = "meta-llama/Llama-3-8b-chat-hf"
                    else:
                        model_name = "unknown"
                    
                    # Ensure all necessary metadata fields are populated
                    state["system_metadata"]["model_used"] = model_name
                    state["system_metadata"]["provider"] = provider_name
                    state["system_metadata"]["embedding_model"] = "sentence-transformers/all-mpnet-base-v2"
                    if "timestamp" not in state["system_metadata"] or not state["system_metadata"]["timestamp"]:
                        state["system_metadata"]["timestamp"] = datetime.datetime.utcnow().isoformat()
                    
                    print(f"Updated metadata after LLM invocation: {state['system_metadata']}")
                
                return response
            except Exception as e:
                print(f"Error with {provider_name} LLM: {str(e)}")
                errors.append(f"{provider_name}: {str(e)}")
        
        # If all providers failed, raise an exception with details
        raise Exception(f"All LLM providers failed: {'; '.join(errors)}")

# --- Initialize embeddings model ---
def get_embeddings_model():
    """Initialize and return a remote embeddings model"""
    try:
        # Using HuggingFace's all-MiniLM-L6-v2 for lightweight but effective embeddings
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder="./hf_cache"
        )
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embeddings: {e}")
        # Fall back to default HuggingFace embeddings if specific model fails
        try:
            return HuggingFaceEmbeddings()
        except Exception as fallback_error:
            logger.critical(f"Critical: Failed to initialize fallback embeddings: {fallback_error}")
            raise

# --- Determine PDF File Paths Relative to this Script ---
# Assumes main.py is inside the 'app' directory, and PDFs are siblings
script_dir = os.path.dirname(__file__) # Gets the directory where main.py is located
pdf_file_paths = [
    os.path.join(script_dir, "hbson-student-handbook.pdf"),
    os.path.join(script_dir, "StudentHandbook.pdf"),
    os.path.join(script_dir, "Student_Handbook (1).pdf") # Ensure this filename is exact
]
print(f"Looking for PDFs in: the path")
print(f"PDF paths to load in paths")


# --- Opik/Comet Configuration ---
opik_tracer = None
if COMET_API_KEY:
    try:
        opik.configure(api_key=COMET_API_KEY, use_local=False) # Configure with key
        print("Opik configured to use Comet.ml")
        # Add tags to identify these runs in Comet/Opik
        opik_tracer = OpikTracer(
            project_name=OPIK_PROJECT_NAME, # Pass project name here
            tags=["local-run", "cuny-suny-agent-v4-opik"]
            )
        print(f"OpikTracer initialized for project OPIK PROJECT NAME ")
    except Exception as e:
        print(f"WARNING: Failed to configure Opik or initialize OpikTracer")
        opik_tracer = None # Ensure it's None if setup fails
else:
    print("Opik tracing disabled (COMET_API_KEY not found).")
# --- End Opik Configuration ---


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize LLM and Tools (With Enhanced DDG Search Tool) ---
try:
    # Initialize LLM provider and embeddings
    llm_provider = LLMProvider()
    embeddings_model = get_embeddings_model()
    
    logger.info(f"Successfully initialized {len(llm_provider.llm_providers)} LLM providers")
    logger.info(f"Embeddings model initialized: {type(embeddings_model).__name__}")

    # --- Start: Enhanced Search Tool Initialization ---
    try:
        wrapper = DuckDuckGoSearchAPIWrapper(
            region="us-en", max_results=10, safesearch='moderate'
         )
        search_tool = DuckDuckGoSearchRun(api_wrapper=wrapper)
        print("DuckDuckGo Search Tool initialized with custom wrapper (max_results=10, region=us-en).")
    except Exception as e:
         logger.error(f"Failed to initialize custom DuckDuckGo Search Tool: {e}. Falling back to default.")
         search_tool = DuckDuckGoSearchRun()
    # --- End: Enhanced Search Tool Initialization ---

    print("LLM Provider and Embeddings initialized.")

except Exception as e:
    logger.critical(f"CRITICAL: Error initializing AI components: {e}")
    logger.critical("Ensure API keys are valid and APIs are responsive.")
    raise SystemExit(f"Failed to initialize core AI components: {e}")

# --- MongoDB setup ---
mongo_client = None
db = None
prof_collection = None
school_collection = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, appName="localAIAsst")
        mongo_client.admin.command('ping') # Verify connection
        print("MongoDB connection successful.")
        db = mongo_client["ratemyprofessors"] # Consider making DB name configurable
        prof_collection = db["suny_professors"]
        school_collection = db["suny_schools"]
    except Exception as e:
        logger.error(f"Error connecting to MongoDB at {MONGO_URI}: {e}")
        logger.error(f"Check your MONGO_URI in the .env file.")
        # Reset variables if connection fails
        mongo_client = None
        db = None
        prof_collection = None
        school_collection = None
else:
    logger.warning("MongoDB connection skipped as MONGO_URI is not set.")


# --- Agent state definition ---
class AgentState(TypedDict):
    query: str
    original_query: Optional[str]
    agent_type: Optional[str]
    knowledge_info: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]] # Can store structured search results if needed
    response: str
    sources: Dict[str, Any]

# --- Custom function to convert ObjectId to string ---
def convert_objectid(data):
    if isinstance(data, dict):
        return {k: convert_objectid(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_objectid(item) for item in data]
    elif isinstance(data, ObjectId):
        return str(data)
    return data

# --- Knowledge Base Manager ---
class KnowledgeBaseManager:
    def __init__(self, pdf_paths: List[str]):
        self.embeddings = get_embeddings_model()
        self.vector_store = self._load_pdfs(pdf_paths)
        if self.vector_store is None:
             logger.error("Knowledge Base initialization failed. KB functionality will be unavailable.")

    def _load_pdfs(self, pdf_paths):
        docs = []
        # Check existence using the absolute paths derived earlier
        valid_paths = [p for p in pdf_paths if os.path.exists(p)]
        if not valid_paths:
             logger.error(f"No valid PDF files found at the specified paths: {pdf_paths}")
             return None
        logger.info(f"Loading PDFs from: {valid_paths}")
        for path in valid_paths:
            try:
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                if not loaded_docs:
                    logger.warning(f"PyPDFLoader returned no documents for: {path}")
                    continue
                # Add source metadata immediately after loading
                for doc in loaded_docs:
                    doc.metadata['source'] = os.path.basename(path) # Add filename as source
                    # Keep existing page number if PyPDFLoader adds it
                docs.extend(loaded_docs)
                logger.info(f"Successfully loaded {len(loaded_docs)} pages from PDF: {path}")
            except Exception as e:
                logger.error(f"Failed to load or parse PDF {path}: {str(e)}", exc_info=True)

        if not docs:
            logger.error("No documents successfully loaded from any valid PDF paths.")
            return None

        try:
            # Adjust chunking strategy if needed
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500, # Slightly smaller chunks might be better for some models
                chunk_overlap=300,
                separators=["\n\n", "\n", ". ", " ", ""],
                keep_separator=False
            )
            split_docs = splitter.split_documents(docs)
            logger.info(f"Split into {len(split_docs)} chunks")
        except Exception as e:
             logger.error(f"Error splitting documents: {e}", exc_info=True)
             return None

        if not split_docs:
             logger.error("Splitting documents resulted in an empty list.")
             return None

        try:
            if not self.embeddings:
                 logger.error("Embeddings model not available for FAISS creation.")
                 return None
            logger.info(f"Creating FAISS index with embeddings: {type(self.embeddings)}")
            vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            logger.info("FAISS vector store created successfully.")
            # Optional: Save the index locally to avoid reprocessing PDFs every time
            # try:
            #     vectorstore.save_local(os.path.join(script_dir, "faiss_index"))
            #     logger.info("FAISS index saved locally.")
            # except Exception as save_e:
            #     logger.error(f"Could not save FAISS index: {save_e}")
            return vectorstore
        except Exception as e:
            logger.error(f"Error creating FAISS index: {e}", exc_info=True)
            return None

    def query(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        if not self.vector_store:
            logger.warning("KB query attempted but no vector store available.")
            return []
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search(query, k=k)
            if not docs:
                 logger.info(f"Similarity search for '{query}' returned no documents.")
                 return []
            # Return structured data including metadata (source, page)
            results = []
            for doc in docs:
                # Standardize the format to match what our agent functions expect
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                results.append({
                    "content": content,
                    "metadata": {
                        "source": metadata.get('source', 'Unknown PDF'),
                        "page": metadata.get('page', 'N/A') # PyPDFLoader usually adds 'page'
                    }
                })
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)
            return []

# --- MongoDB Database Access Classes ---
class ProfessorDatabase:
    def __init__(self, collection):
        self.collection = collection

    def query_professor(self, name: str, school: str) -> Dict[str, Any]:
        if self.collection is None:
             logger.error("Professor collection not available (MongoDB connection failed or not configured).")
             return {}
        if not name: # School might be optional if name is unique enough
             logger.warning(f"Query professor called with empty name.")
             return {}
        try:
            # Build query: flexible regex for name, optional regex for school
            query_filter = {
                "data.professor_info.name": {"$regex": re.escape(name), "$options": "i"}
            }
            if school: # Only add school filter if provided
                 query_filter["data.professor_info.school"] = {"$regex": re.escape(school), "$options": "i"}

            logger.info(f"Executing Professor DB query: {query_filter}")
            result = self.collection.find_one(query_filter)

            if result:
                 logger.info(f"Found professor match for Name: '{name}', School: '{school or 'Any'}'")
                 # Return only the 'data' part, converted
                 return convert_objectid(result.get("data", {}))
            else:
                 logger.info(f"No professor match found in DB for Name: '{name}', School: '{school or 'Any'}'")
                 return {}
        except Exception as e:
             logger.error(f"Error querying professor DB: {e}", exc_info=True)
             return {}
             
    def query_professor_by_full_name(self, first_name: str, last_name: str, school: str = None) -> Dict[str, Any]:
        """
        Query for a professor by first and last name, with optional school filter
        Returns the full professor data if found, empty dict otherwise
        """
        if self.collection is None:
            logger.error("Professor collection not available (MongoDB connection failed or not configured).")
            return {}
            
        try:
            # Create a regex pattern that matches either the full name or first/last name separately
            full_name_pattern = f"{first_name}.*{last_name}"
            
            # Build query with flexible name matching and optional school
            query_filter = {
                "data.professor_info.name": {"$regex": full_name_pattern, "$options": "i"}
            }
            
            if school: # Only add school filter if provided
                query_filter["data.professor_info.school"] = {"$regex": re.escape(school), "$options": "i"}
                
            logger.info(f"Executing Professor DB query by full name: {query_filter}")
            result = self.collection.find_one(query_filter)
            
            if result:
                logger.info(f"Found professor match for Name: '{first_name} {last_name}', School: '{school or 'Any'}'")
                # Return only the 'data' part, converted
                return convert_objectid(result.get("data", {}))
            else:
                logger.info(f"No professor match found in DB for Name: '{first_name} {last_name}', School: '{school or 'Any'}'")
                # Try a more permissive search if exact full name doesn't match
                return self.query_professor(f"{first_name} {last_name}", school)
        except Exception as e:
            logger.error(f"Error querying professor DB by full name: {e}", exc_info=True)
            return {}
            
    def search_professors_by_name(self, name_part: str) -> List[Dict[str, Any]]:
        """
        Search for professors by partial name match
        Returns a list of matching professors
        """
        if self.collection is None:
            logger.error("Professor collection not available (MongoDB connection failed or not configured).")
            return []
            
        try:
            # Build query with flexible name matching
            query_filter = {
                "data.professor_info.name": {"$regex": re.escape(name_part), "$options": "i"}
            }
            
            logger.info(f"Searching professors by name part: {query_filter}")
            results = list(self.collection.find(query_filter).limit(5))  # Limit to prevent large result sets
            
            processed_results = []
            for result in results:
                # Extract just the professor data, not the full document
                prof_data = result.get("data", {}).get("professor_info", {})
                if prof_data:
                    processed_results.append(convert_objectid(prof_data))
            
            return processed_results
        except Exception as e:
            logger.error(f"Error searching professors by name: {e}", exc_info=True)
            return []
            
    def search_professors(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for professors based on any matching terms in the query
        Returns a list of matching professors
        """
        if self.collection is None:
            logger.error("Professor collection not available (MongoDB connection failed or not configured).")
            return []
            
        try:
            # Extract potential name parts and department/subject terms from the query
            query_terms = query.lower().strip().split()
            
            # Simple search by any term
            results = []
            for term in query_terms:
                if len(term) < 3:  # Skip very short terms
                    continue
                    
                # Build query with flexible matching on name or department
                query_filter = {
                    "$or": [
                        {"data.professor_info.name": {"$regex": re.escape(term), "$options": "i"}},
                        {"data.professor_info.department": {"$regex": re.escape(term), "$options": "i"}}
                    ]
                }
                
                logger.info(f"Searching professors by term '{term}': {query_filter}")
                batch_results = list(self.collection.find(query_filter).limit(5))
                
                for result in batch_results:
                    # Extract just the professor data, not the full document
                    prof_data = result.get("data", {}).get("professor_info", {})
                    if prof_data and prof_data not in results:  # Avoid duplicates
                        results.append(convert_objectid(prof_data))
            
            return results[:5]  # Limit results to top 5
        except Exception as e:
            logger.error(f"Error in general professor search: {e}", exc_info=True)
            return []

class SchoolDatabase:
    def __init__(self, collection):
        self.collection = collection

    def query_school(self, name: str) -> Dict[str, Any]:
        """Query for a specific school by name"""
        if self.collection is None:
            logger.error("School collection not available (MongoDB connection failed or not configured).")
            return {}
        
        try:
            # Create a case-insensitive regex query for partial name matches
            query_filter = {"name": {"$regex": name, "$options": "i"}}
            logger.info(f"Executing School DB query: {query_filter}")
            result = self.collection.find_one(query_filter)
            
            if result:
                # Convert ObjectId to string for JSON serialization
                return convert_objectid(result)
            else:
                logger.info(f"No school found with name: {name}")
                return {}
        except Exception as e:
            logger.error(f"Error querying school database: {e}")
            return {}
    
    def list_schools(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List all schools up to the limit"""
        if self.collection is None:
            logger.error("School collection not available (MongoDB connection failed or not configured).")
            return []
        
        try:
            # Find all schools, sorted by name
            # Only return name, category, and URLs to keep it concise
            projection = {"name": 1, "category": 1, "urls": 1}
            results = list(self.collection.find({}, projection).sort("name", 1).limit(limit))
            
            # Convert ObjectIds to strings for JSON serialization
            return convert_objectid(results)
        except Exception as e:
            logger.error(f"Error listing schools from database: {e}")
            return []

# --- Initialize resources ---
# Check if FAISS index exists locally first (optional optimization)
# faiss_index_path = os.path.join(script_dir, "faiss_index")
# if os.path.exists(faiss_index_path):
#     try:
#         kb = KnowledgeBaseManager([]) # Init empty first
#         kb.vector_store = FAISS.load_local(faiss_index_path, embeddings_model, allow_dangerous_deserialization=True)
#         logger.info(f"Loaded existing FAISS index from {faiss_index_path}")
#     except Exception as load_e:
#         logger.error(f"Failed to load local FAISS index: {load_e}. Rebuilding from PDFs.")
#         kb = KnowledgeBaseManager(pdf_file_paths)
# else:
#     logger.info("No local FAISS index found. Building from PDFs...")
kb = KnowledgeBaseManager(pdf_file_paths) # Always build for now, simpler

prof_db = ProfessorDatabase(prof_collection)
school_db = SchoolDatabase(school_collection)

# --- Agent Node Functions ---

def determine_agent_type(state: AgentState) -> AgentState:
    """Route queries to appropriate agent based on content."""
    query = state["query"].lower()
    # Clear previous agent results from state
    state["knowledge_info"] = []
    state["search_results"] = [] # Changed to list for consistency
    state["response"] = ""
    state["sources"] = {}
    
    # Add system metadata to track model usage
    state["system_metadata"] = {
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "model_used": None,  # Will be populated by the specific agent
        "provider": None,    # Will be populated by the specific agent
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Store original query
    state["original_query"] = state["query"]
    
    # Check for basic greeting patterns first
    greeting_patterns = [
        "hello", "hi", "hey", "greetings", "what's up", "how are you", 
        "how do you do", "how's it going", "good morning", "good afternoon", 
        "good evening", "howdy"
    ]
    
    if any(pattern in query for pattern in greeting_patterns) and len(query.split()) <= 8:
        # This is a simple greeting, respond directly without complex search
        responses = [
            "Hello! I'm the CUNY/SUNY AI Assistant. How can I help you with college-related questions today?",
            "Hi there! I'm here to help with information about CUNY and SUNY schools. What would you like to know?",
            "Hello! I'm your CUNY/SUNY AI Assistant. I can help with transfers, professors, school recommendations, and more. What can I assist you with?",
            "Greetings! I'm the AI Assistant for CUNY/SUNY students. I'm doing well, thanks for asking! How can I help you today?"
        ]
        # Use consistent response or pick randomly for variety
        state["response"] = responses[0]  # For consistency, always use the first one
        state["agent_type"] = "greeting"
        # Set metadata for greeting (no model is actually used)
        state["system_metadata"]["model_used"] = "direct_response"
        state["system_metadata"]["provider"] = "system"
        return state

    # Check if the query is not related to SUNY/CUNY
    off_topic_patterns = [
        "weather", "time in", "what time", "stock", "sports", "game", 
        "movie", "recipe", "news", "bitcoin", "crypto", "current time",
        "what's the time", "tell me the time", "time now", "president"
    ]
    
    # Only consider off-topic if it doesn't also contain SUNY/CUNY related terms
    education_terms = ["cuny", "suny", "college", "university", "school", "campus", "professor", 
                      "class", "course", "degree", "major", "student", "semester", "admission", 
                      "application", "program", "transfer"]
    
    if any(pattern in query for pattern in off_topic_patterns) and not any(term in query for term in education_terms):
        # This is an off-topic query, respond with a reminder of what this assistant is for
        state["response"] = "I'm designed specifically to help with questions about SUNY and CUNY colleges and universities. I can assist with information about programs, transfers, professors, admissions, campus life, and other educational topics related to these institutions. For other types of information like current time, weather, or general questions, please use a general search engine or virtual assistant. How can I help you with SUNY/CUNY related information today?"
        state["agent_type"] = "off_topic"
        # Set metadata
        state["system_metadata"]["model_used"] = "direct_response"
        state["system_metadata"]["provider"] = "system"
        
        # Return off-topic response directly
        return QueryResponse(
            response=state["response"],
            agent_type=state["agent_type"],
            sources=state["sources"],
            original_query=state["original_query"],
            system_metadata=SystemMetadata(**state["system_metadata"])
        )

    # Continue with regular routing for non-greeting, on-topic queries
    if "transfer" in query or "switch school" in query:
        state["agent_type"] = "transfer"
    elif "recommend" in query or "best school" in query or "major" in query or "career" in query or "program" in query:
        state["agent_type"] = "recommendation"
    elif "professor" in query or "prof" in query or "rate" in query or "review" in query or "instructor" in query:
        # Basic check for professor query structure
         if " at " in query or re.search(r'\b(at|from)\b', query, re.IGNORECASE):
             state["agent_type"] = "professor"
         else:
              # Could be general info, default to browser
              logger.info("Query contains 'professor' but lacks clear structure (e.g., 'at School'). Routing to browser.")
              state["agent_type"] = "browser"

    else: # Default to browser for general queries
        state["agent_type"] = "browser"

    logger.info(f"Routing query to agent: {state['agent_type']}")
    return state

def transfer_agent(state: AgentState) -> AgentState:
    """Agent that handles transfer-related queries"""
    query = state["query"]
    
    # Get knowledge base info
    kb_info = []
    try:
        if kb:
            kb_info = kb.query(query, k=3) # Get top 3 results
    except Exception as e:
        logger.error(f"Error querying knowledge base for transfer info: {e}")
    
    # Format KB results
    info_text = ""
    kb_sources = []
    if kb_info:
        for i, doc in enumerate(kb_info):
            # The docs are already dictionaries with content and metadata
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', 'N/A')
            kb_sources.append({"content": doc.get('content', ''), "metadata": {"source": source, "page": page}})
            info_text += f"\nHandbook excerpt {i+1} (Source: {source}, Page: {page}):\n{doc.get('content', '')}\n"
    else:
        info_text = "No transfer-specific information found in handbooks."
    
    # Get web search info
    search_text = ""
    search_results = []
    try:
        specific_query = f"CUNY SUNY college transfer requirements process {query}"
        search_result = search_tool.run(specific_query)
        if search_result:
            search_results.append({"source": "DuckDuckGo", "content": search_result})
            search_text = f"Web search results for transfer information:\n{search_result}"
    except Exception as e:
        logger.error(f"Error with transfer web search: {e}")
        search_results.append({"source": "DuckDuckGo", "error": str(e)})
        search_text = f"Note: Web search failed with error: {e}"
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are a transfer advisor for CUNY/SUNY students. Answer the following transfer-related question based on the information provided.

Handbook Information:
{info}

Web Search Results:
{search}

Transfer Question: {query}

In your response:
1. Be specific about transfer requirements, deadlines, and processes
2. Mention any potential issues or special considerations
3. Suggest next steps the student should take
4. Specify which colleges/programs information applies to when relevant
5. Note if any information is missing or needs verification

Answer:""",
        input_variables=["info", "search", "query"]
    )
    
    # Invoke LLM
    try:
        response = llm_provider.invoke(prompt.format(info=info_text, search=search_text, query=query), state=state)
        state["response"] = str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"Error invoking LLM for transfer query: {e}")
        state["response"] = f"I apologize, but I encountered an error while processing your transfer question: {str(e)}"
    
    # Store sources
    state["sources"] = {
        "knowledge_base": kb_sources if kb_sources else None,
        "search": search_results if search_results else None
    }
    
    state["agent_type"] = "transfer"
    return state

def recommendation_agent(state: AgentState) -> AgentState:
    """Agent that handles school recommendation queries"""
    query = state["query"]
    
    # Get knowledge base info first
    kb_info = []
    try:
        if kb:
            kb_info = kb.query(query, k=3)
    except Exception as e:
        logger.error(f"Error querying knowledge base for recommendation info: {e}")
    
    # Format KB results
    info_text = ""
    kb_sources = []
    if kb_info:
        for i, doc in enumerate(kb_info):
            # The docs are already dictionaries with content and metadata
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', 'N/A')
            kb_sources.append({"content": doc.get('content', ''), "metadata": {"source": source, "page": page}})
            info_text += f"\nHandbook excerpt {i+1} (Source: {source}, Page: {page}):\n{doc.get('content', '')}\n"
    else:
        info_text = "No specific school recommendation information found in handbooks."
    
    # Get web search info
    search_text = ""
    search_results = []
    try:
        specific_query = f"CUNY SUNY college recommendation for students interested in {query}"
        search_result = search_tool.run(specific_query)
        if search_result:
            search_results.append({"source": "DuckDuckGo", "content": search_result})
            search_text = f"Web search results:\n{search_result}"
    except Exception as e:
        logger.error(f"Error with recommendation web search: {e}")
        search_results.append({"source": "DuckDuckGo", "error": str(e)})
        search_text = f"Note: Web search failed with error: {e}"
    
    # Get school data from database if available
    schools_text = ""
    schools_data = []
    if school_collection is not None:
        try:
            schools = SchoolDatabase(school_collection).list_schools(limit=10)
            if schools:
                schools_data = schools
                schools_text = "Schools in database:\n"
                for school in schools:
                    schools_text += f"- {school.get('name', 'Unknown')} ({school.get('category', 'Unknown')})\n"
        except Exception as e:
            logger.error(f"Error getting schools from database: {e}")
    
    # Create prompt
    prompt = PromptTemplate(
        template="""You are a school recommendation assistant for CUNY/SUNY students. Answer the following question based on the information provided.

Handbook Information:
{info}

School Information:
{schools}

Web Search Results:
{search}

Student Question: {query}

In your response:
1. Recommend specific schools/programs that match the student's interests
2. Mention key features, strengths, and requirements of each recommended school
3. Consider location, size, program offerings, and specializations 
4. Suggest resources for further research
5. Be honest about limitations in your knowledge

Answer:""",
        input_variables=["info", "schools", "search", "query"]
    )
    
    # Invoke LLM
    try:
        response = llm_provider.invoke(prompt.format(info=info_text, search=search_text, schools=schools_text, query=query), state=state)
        state["response"] = str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"Error invoking LLM for recommendation query: {e}")
        state["response"] = f"I apologize, but I encountered an error while processing your school recommendation question: {str(e)}"
    
    # Store sources
    state["sources"] = {
        "knowledge_base": kb_sources if kb_sources else None,
        "search": search_results if search_results else None,
        "school_db": schools_data if schools_data else None
    }
    
    state["agent_type"] = "recommendation"
    return state

def professor_agent(state: AgentState) -> AgentState:
    """Agent that handles professor information queries"""
    query = state["query"]
    
    # Check if this is a structured query from the professor endpoint
    is_structured_query = "professor_details" in state
    
    # Get professor info from database if available
    prof_text = ""
    prof_data = None  # Changed from list to None (will be a dict)
    
    if prof_collection is not None:
        try:
            professor_db = ProfessorDatabase(prof_collection)
            
            # Handle structured query from the professor endpoint
            if is_structured_query:
                details = state["professor_details"]
                first_name = details["first_name"]
                last_name = details["last_name"]
                college_name = details["college_name"]
                
                logger.info(f"Processing structured professor query for {first_name} {last_name} at {college_name}")
                
                # Look up the professor using the structured data
                professor = professor_db.query_professor_by_full_name(first_name, last_name, college_name)
                
                if professor:
                    # Format professor details
                    prof_info = professor.get('professor_info', {})
                    prof_reviews = professor.get('reviews', [])
                    
                    # Store for source attribution - proper format for ProfessorDBSource
                    prof_data = {
                        "professor_info": prof_info,
                        "reviews": prof_reviews
                    }
                    
                    # Build a detailed text summary
                    prof_text = "Professor information in database:\n"
                    prof_text += f"- Name: {prof_info.get('name', f'{first_name} {last_name}')}\n"
                    prof_text += f"- School: {prof_info.get('school', college_name)}\n"
                    
                    # Add department and courses if available
                    dept = prof_info.get('department', 'Unknown Department')
                    prof_text += f"- Department: {dept}\n"
                    
                    if 'courses' in prof_info:
                        courses = ", ".join(prof_info.get('courses', ['No courses listed']))
                        prof_text += f"- Courses: {courses}\n"
                    
                    # Add ratings and statistics
                    rating = prof_info.get('rating', 'No rating available')
                    prof_text += f"- Overall Rating: {rating}\n"
                    
                    # Include difficulty if available
                    if 'difficulty' in prof_info:
                        prof_text += f"- Difficulty: {prof_info.get('difficulty', 'N/A')}\n"
                    
                    # Add would take again percentage if available
                    if 'would_take_again' in prof_info:
                        prof_text += f"- Would Take Again: {prof_info.get('would_take_again', 'N/A')}%\n"
                    
                    # Add contact info if available
                    if 'email' in prof_info:
                        prof_text += f"- Email: {prof_info.get('email', 'N/A')}\n"
                    
                    if 'office_hours' in prof_info:
                        prof_text += f"- Office Hours: {prof_info.get('office_hours', 'N/A')}\n"
                    
                    # Add review summary
                    if prof_reviews:
                        prof_text += "\nStudent Reviews:\n"
                        # Limit to 3 reviews for readability
                        for i, review in enumerate(prof_reviews[:3]):
                            rating = review.get('rating', 'N/A')
                            date = review.get('date', 'Unknown date')
                            comment = review.get('comment', 'No comment provided')
                            course = review.get('course', 'Unknown course')
                            
                            prof_text += f"Review {i+1}:\n"
                            prof_text += f"- Rating: {rating}/5\n"
                            prof_text += f"- Course: {course}\n"
                            prof_text += f"- Date: {date}\n"
                            prof_text += f"- Comment: {comment}\n\n"
                else:
                    prof_text = f"No information found in our database for Professor {first_name} {last_name} at {college_name}."
                    # Try a more general search by last name
                    fallback_results = professor_db.search_professors_by_name(last_name)
                    if fallback_results:
                        # Format for ProfessorDBSource - use the first result as main professor info
                        # and include all results in a special "similar_professors" field
                        prof_data = {
                            "professor_info": {
                                "name": f"{first_name} {last_name}",
                                "school": college_name,
                                "similar_professors": fallback_results  # Put the list in a custom field
                            },
                            "reviews": []
                        }
                        
                        prof_text += "\n\nHowever, we found these professors with similar names:\n"
                        for prof in fallback_results:
                            prof_text += f"- {prof.get('name', 'Unknown')}"
                            if 'school' in prof:
                                prof_text += f" at {prof.get('school', 'Unknown School')}"
                            prof_text += "\n"
            
            # Handle unstructured query from the general query endpoint
            else:
                # Try to identify if a specific name is being requested
                name_pattern = r'(?:professor|prof\.?|dr\.?)\s+([a-z]+)'
                name_match = re.search(name_pattern, query.lower())
                
                if name_match:
                    # Search for the specific professor name
                    last_name = name_match.group(1)
                    professors = professor_db.search_professors_by_name(last_name)
                else:
                    # More general search based on departments or keywords
                    professors = professor_db.search_professors(query)
                    
                if professors:
                    # Format for ProfessorDBSource - use the first professor as the main info
                    # and include the full list as a special field
                    prof_data = {
                        "professor_info": professors[0],  # Use first professor as main info
                        "reviews": [],  # No reviews in this case
                        "all_matches": professors  # Store all matches in a custom field
                    }
                    
                    prof_text = "Professor information in database:\n"
                    for prof in professors:
                        dept = prof.get('department', 'Unknown Department')
                        courses = ", ".join(prof.get('courses', ['No courses listed']))
                        rating = prof.get('rating', 'No rating available')
                        office = prof.get('office_hours', 'No office hours listed')
                        email = prof.get('email', 'No email listed')
                        
                        prof_text += f"- {prof.get('name', 'Unknown')}\n"
                        prof_text += f"  Department: {dept}\n"
                        prof_text += f"  Courses: {courses}\n"
                        prof_text += f"  Rating: {rating}\n"
                        prof_text += f"  Office Hours: {office}\n"
                        prof_text += f"  Email: {email}\n\n"
        except Exception as e:
            logger.error(f"Error getting professor data from database: {e}")
    
    # Get knowledge base info
    kb_info = []
    try:
        if kb:
            kb_info = kb.query(query, k=3)
    except Exception as e:
        logger.error(f"Error querying knowledge base for professor info: {e}")
    
    # Format KB results
    info_text = ""
    kb_sources = []
    if kb_info:
        for i, doc in enumerate(kb_info):
            # The docs are already dictionaries with content and metadata
            source = doc.get('metadata', {}).get('source', 'Unknown')
            page = doc.get('metadata', {}).get('page', 'N/A')
            kb_sources.append({"content": doc.get('content', ''), "metadata": {"source": source, "page": page}})
            info_text += f"\nHandbook excerpt {i+1} (Source: {source}, Page: {page}):\n{doc.get('content', '')}\n"
    else:
        info_text = "No specific professor information found in handbooks."
    
    # Get web search info
    search_text = ""
    search_results = []
    try:
        # Customize the search query based on whether it's structured or not
        if is_structured_query:
            details = state["professor_details"]
            specific_question = details["specific_question"]
            search_query = f"Professor {details['first_name']} {details['last_name']} {details['college_name']} {specific_question}"
        else:
            search_query = f"CUNY SUNY professor information {query}"
            
        search_result = search_tool.run(search_query)
        if search_result:
            search_results.append({"source": "DuckDuckGo", "content": search_result})
            search_text = f"Web search results:\n{search_result}"
    except Exception as e:
        logger.error(f"Error with professor web search: {e}")
        search_results.append({"source": "DuckDuckGo", "error": str(e)})
        search_text = f"Note: Web search failed with error: {e}"
    
    # Customize prompt based on whether it's a structured query
    template = """You are a university assistant providing information about professors. 
Answer the following question based on the information provided.

Professor Database Information:
{professors}

Handbook Information:
{info}

Web Search Results:
{search}

"""

    # Add specific instructions for structured queries
    if is_structured_query:
        details = state["professor_details"]
        specific_question = details["specific_question"]
        template += f"""Student's specific question about Professor {details['first_name']} {details['last_name']} at {details['college_name']}:
{specific_question}

In your response:
1. Directly address the student's specific question about the professor
2. Provide accurate information based on the database and search results
3. Include relevant details about teaching style, courses, ratings, and reviews if available
4. If information is limited, acknowledge this and provide what is available
5. Maintain a helpful, objective tone when discussing professor ratings and reviews

Answer:"""
    else:
        template += """Student Question: {query}

In your response:
1. Provide accurate information about the professor(s) mentioned or relevant to the query
2. Include details about teaching areas, office hours, and contact information if available
3. If course-specific information was requested, provide that context
4. Be honest about limitations in your knowledge

Answer:"""
    
    # Create the prompt
    prompt = PromptTemplate(
        template=template,
        input_variables=["professors", "info", "search", "query"] if not is_structured_query else ["professors", "info", "search"]
    )
    
    # Invoke LLM
    try:
        if is_structured_query:
            response = llm_provider.invoke(prompt.format(professors=prof_text, info=info_text, search=search_text), state=state)
        else:
            response = llm_provider.invoke(prompt.format(professors=prof_text, info=info_text, search=search_text, query=query), state=state)
            
        state["response"] = str(response.content) if hasattr(response, 'content') else str(response)
    except Exception as e:
        logger.error(f"Error invoking LLM for professor query: {e}")
        state["response"] = f"I apologize, but I encountered an error while processing your professor information question: {str(e)}"
    
    # Store sources - ensure professor_db is either a dict or None, not a list
    state["sources"] = {
        "professor_db": prof_data,  # Already formatted as a dictionary for ProfessorDBSource
        "knowledge_base": kb_sources if kb_sources else None,
        "search": search_results if search_results else None
    }
    
    state["agent_type"] = "professor"
    return state


# --- Enhanced Browser Agent (Checks KB + Web) ---
def browser_agent(state: AgentState) -> AgentState:
    """Agent that handles general questions through a mix of KB and web search"""
    query = state["query"]
    
    # First, check our knowledge base for relevant documents
    kb_info = []
    try:
        if kb:
            kb_info = kb.query(query, k=3)  # Get top 3 most relevant chunks
    except Exception as e:
        logger.error(f"Error querying knowledge base: {e}")
    
    # Format knowledge base info for prompt
    kb_info_text = ""
    kb_sources = []
    if kb_info:
        for i, doc in enumerate(kb_info):
            # Extract source and page from metadata if available
            # The docs are already dictionaries with content and metadata
            source = doc.get('metadata', {}).get('source', 'N/A')
            page = doc.get('metadata', {}).get('page', 'N/A')
            kb_sources.append({"content": doc.get('content', ''), "metadata": {"source": source, "page": page}})
            kb_info_text += f"\nHandbook excerpt {i+1} (Source: {source}, Page: {page}):\n{doc.get('content', '')}\n"
    
    # Always perform web search with a SUNY/CUNY focused query to supplement knowledge
    search_results = []
    search_text = ""
    try:
        # Add SUNY/CUNY keywords to make sure search is relevant to our domain
        focused_query = f"SUNY CUNY {query} official information"
        search_result = search_tool.run(focused_query)
        if search_result:
            search_results.append({"source": "DuckDuckGo", "content": search_result})
            search_text = f"Web search results:\n{search_result}"
    except Exception as e:
        logger.error(f"Error with web search: {e}")
        search_results.append({"source": "DuckDuckGo", "error": str(e)})
        search_text = f"Note: Web search failed with error: {e}"

    # Construct the prompt
    prompt = PromptTemplate(
        template="""You are a helpful SUNY/CUNY college assistant. You provide clear, direct answers about SUNY and CUNY colleges.

Information from the handbook:
{info}

Web search results:
{search}

Question: {query}

Important instructions:
1. Be direct and answer the question clearly right at the beginning
2. Start with the most relevant information the user is asking for
3. Mention specific details like times, locations, or dates when available
4. If you don't have the exact information, clearly state what you do know and what's missing
5. Keep your answer focused and concise
6. Only discuss information related to SUNY and CUNY schools

Answer:""",
        input_variables=["info", "search", "query"]
    )

    # Invoke LLM
    try:
        # Log state before invoking LLM
        logger.info(f"Browser agent metadata before LLM: {state.get('system_metadata')}")
        
        response = llm_provider.invoke(prompt.format(info=kb_info_text, search=search_text, query=query), state=state)
        state["response"] = str(response.content) if hasattr(response, 'content') else str(response)
        
        # Log state after invoking LLM
        logger.info(f"Browser agent metadata after LLM: {state.get('system_metadata')}")
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}")
        state["response"] = f"I apologize, but I encountered an error while processing your request: {str(e)}"
        # Set fallback metadata in case of error
        state["system_metadata"]["model_used"] = "error_fallback"
        state["system_metadata"]["provider"] = "system"

    # Store sources for citation/display
    state["sources"] = {
        "knowledge_base": kb_sources if kb_sources else None,
        "search": search_results if search_results else None
    }
    
    state["agent_type"] = "browser"
    return state


# --- Workflow setup ---
workflow = StateGraph(AgentState)
workflow.add_node("route", determine_agent_type)
workflow.add_node("transfer", transfer_agent)
workflow.add_node("recommendation", recommendation_agent)
workflow.add_node("professor", professor_agent) # This node now includes DB check and potential fallback
workflow.add_node("browser", browser_agent) # General browser node

workflow.set_entry_point("route")

# Conditional Edges: Route to the appropriate agent
workflow.add_conditional_edges(
    "route",
    lambda s: s["agent_type"],
    {
        "greeting": END,  # Greeting responses end immediately
        "transfer": "transfer",
        "recommendation": "recommendation",
        "professor": "professor", # Professor agent handles DB lookup AND fallback decision
        "browser": "browser",
        "browser_fallback_professor": "browser" # Route professor fallback explicitly to browser node
     }
)

# End edges: All processing nodes lead to END
workflow.add_edge("transfer", END)
workflow.add_edge("recommendation", END)
workflow.add_edge("professor", END) # Professor agent ends the graph whether it finds in DB or calls browser internally
workflow.add_edge("browser", END)


# Compile the graph
graph = None
try:
    graph = workflow.compile()
    print("--- LangGraph graph compiled successfully ---")
except Exception as e:
    logger.critical(f"CRITICAL: Error compiling LangGraph workflow: {e}", exc_info=True)
    # Decide how to handle this - maybe exit or run FastAPI without the graph?
    # For now, we'll let FastAPI start but the endpoint will fail.
    print("--- LangGraph compilation FAILED ---")


# --- FastAPI Application Setup ---
app = FastAPI(
    title="CUNY/SUNY AI Assistant",
    description="API endpoint for the LangGraph powered CUNY/SUNY AI Assistant",
    version="1.0.0"
)

# --- Pydantic Models for API ---
class QueryRequest(BaseModel):
    query: str
    user_id: Optional[str] = None # Optional field for user tracking

class SourceMetadata(BaseModel):
    source: Optional[str] = None
    page: Optional[Any] = None # Page can be int or str 'N/A'

class KnowledgeSource(BaseModel):
    content: Optional[str] = None
    metadata: Optional[SourceMetadata] = None

class SearchSource(BaseModel):
    source: Optional[str] = None # e.g., "DuckDuckGo"
    content: Optional[str] = None
    error: Optional[str] = None

class ProfessorDBSource(BaseModel):
    # Reflect the structure of your professor data
    professor_info: Optional[Dict[str, Any]] = None
    reviews: Optional[List[Dict[str, Any]]] = None
    # Add other top-level keys from your DB structure if necessary

class SchoolDBSource(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None
    urls: Optional[Any] = None # Can be dict or list depending on your data


class ResponseSources(BaseModel):
    knowledge_base: Optional[List[KnowledgeSource]] = None
    search: Optional[List[SearchSource]] = None
    professor_db: Optional[ProfessorDBSource] = None
    school_db: Optional[List[SchoolDBSource]] = None


class SystemMetadata(BaseModel):
    model_used: Optional[str] = None
    embedding_model: Optional[str] = None
    provider: Optional[str] = None
    timestamp: Optional[str] = None

class QueryResponse(BaseModel):
    agent_type: Optional[str] = None
    response: str
    sources: Optional[ResponseSources] = None # Use the detailed sources model
    original_query: Optional[str] = None
    system_metadata: Optional[SystemMetadata] = None # Add system metadata

# Add a new Pydantic model for professor-specific queries after the existing model definitions
class ProfessorRequest(BaseModel):
    first_name: str
    last_name: str
    college_name: str
    question: str
    user_id: Optional[str] = None  # Optional field for user tracking

# --- Content Safety Filter ---
class ContentSafetyFilter:
    def __init__(self, llm_provider):
        self.llm_provider = llm_provider
        self.policy = policies.policy
    
    def check_query(self, query: str) -> tuple[bool, str]:
        """
        Check if the incoming query complies with safety policies.
        Returns: (is_safe, explanation)
        """
        if not query or query.strip() == "":
            return True, "Empty query is safe."
            
        # Skip safety check for simple greetings
        greeting_patterns = [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^greetings\b', r'^howdy\b',
            r'^good morning\b', r'^good afternoon\b', r'^good evening\b'
        ]
        if any(re.search(pattern, query.lower()) for pattern in greeting_patterns) and len(query.split()) <= 5:
            return True, "Simple greeting is safe."
            
        # Create prompt to check if the query is safe
        prompt_template = """You are a content safety filter. Given the following safety policies and user query, determine if the query is safe to process.

SAFETY POLICIES:
{policy}

USER QUERY: 
"{query}"

Does this query violate any of the safety policies? Check each category (O1-O6) and reply with either SAFE or UNSAFE followed by a brief explanation.
If UNSAFE, specify which policy category (e.g., O1, O2) is violated and why.
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["policy", "query"]
        )
        
        try:
            response = self.llm_provider.invoke(prompt.format(policy=self.policy, query=query))
            response_text = str(response.content) if hasattr(response, 'content') else str(response)
            
            # Determine safety based on response
            is_unsafe = "UNSAFE" in response_text.upper()
            
            if is_unsafe:
                logger.warning(f"Query filtered as unsafe: '{query}'")
                return False, response_text
            return True, "Query is safe."
        except Exception as e:
            logger.error(f"Error in safety check: {e}")
            # If safety check fails, assume safe but log the error
            return True, f"Safety check error: {str(e)}"
    
    def filter_response(self, response: str, query: str) -> tuple[str, bool]:
        """
        Filter the AI response to ensure it complies with safety policies.
        Returns: (filtered_response, was_modified)
        """
        if not response or response.strip() == "":
            return response, False
            
        # Create prompt to check if the response is safe
        prompt_template = """You are a content safety filter. Given the following safety policies, user query, and AI response, determine if the response is safe.

SAFETY POLICIES:
{policy}

USER QUERY: 
"{query}"

AI RESPONSE:
"{response}"

Does this response violate any of the safety policies? Check each category (O1-O6) and reply with either SAFE or UNSAFE followed by a brief explanation.
If UNSAFE, specify which policy category (e.g., O1, O2) is violated and why. Then provide a rewritten safe version of the response that addresses the user's query while complying with the policies.
"""
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["policy", "query", "response"]
        )
        
        try:
            filter_response = self.llm_provider.invoke(prompt.format(policy=self.policy, query=query, response=response))
            filter_text = str(filter_response.content) if hasattr(filter_response, 'content') else str(filter_response)
            
            # Check if response was deemed unsafe
            is_unsafe = "UNSAFE" in filter_text.upper()
            
            if is_unsafe:
                logger.warning(f"Response filtered as unsafe for query: '{query}'")
                
                # Extract the safe alternative response if provided
                # Look for patterns like "Safe version:" or "Rewritten response:"
                safe_version_patterns = [
                    r"(?:safe version|rewritten response|here is a safe|alternative response):(.*)",
                    r"(?:safe version|rewritten response|here is a safe|alternative response)(.*)",
                ]
                
                for pattern in safe_version_patterns:
                    match = re.search(pattern, filter_text, re.IGNORECASE | re.DOTALL)
                    if match:
                        safe_response = match.group(1).strip()
                        if safe_response:
                            return safe_response, True
                
                # If no safe version found, return a generic safe response
                return "I apologize, but I cannot provide information on that topic due to content safety policies.", True
            
            return response, False
        except Exception as e:
            logger.error(f"Error in response filtering: {e}")
            # If filtering fails, return the original response but log the error
            return response, False

# --- Initialize Safety Filter after LLM and tools initialization ---
try:
    # Initialize the content safety filter with our LLM provider
    safety_filter = ContentSafetyFilter(llm_provider)
    logger.info("Content Safety Filter initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize Content Safety Filter: {e}")
    safety_filter = None

# --- API Endpoints ---
@app.get("/", include_in_schema=False)
async def root():
    # Redirect to API docs
    return RedirectResponse("/docs")

@app.post("/query", response_model=QueryResponse)
async def route_query(request: QueryRequest):
    """Root route for all queries. Will determine the appropriate agent to handle the query."""
    try:
        # Apply safety filter to incoming query
        if safety_filter:
            is_safe, explanation = safety_filter.check_query(request.query)
            if not is_safe:
                # Return a safe response explaining why the query was rejected
                return QueryResponse(
                    response="I'm sorry, but I cannot respond to that query as it may violate content safety policies. Please ask a different question related to CUNY/SUNY educational topics.",
                    agent_type="safety_filtered",
                    sources=None,
                    original_query=request.query,
                    system_metadata=SystemMetadata(
                        model_used="safety_filter",
                        embedding_model="sentence-transformers/all-mpnet-base-v2",
                        provider="system",
                        timestamp=datetime.datetime.utcnow().isoformat()
                    )
                )
        
        # Initialize state
        state = AgentState(query=request.query, user_id=request.user_id)
        
        # Clear any previous agent results
        state["response"] = None
        state["agent_type"] = None
        state["sources"] = None
        state["system_metadata"] = {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "model_used": None,  # Will be populated during agent execution
            "provider": None,    # Will be populated during agent execution
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store original query 
        state["original_query"] = request.query
        
        # Check for greeting patterns first
        greeting_patterns = [
            r'^hi\b', r'^hello\b', r'^hey\b', r'^greetings\b', r'^howdy\b',
            r'^good morning\b', r'^good afternoon\b', r'^good evening\b',
            r'^how are you\b', r'how\'s it going\b', r'what\'s up\b'
        ]
        
        # Check if the query matches any greeting pattern
        if any(re.search(pattern, request.query.lower()) for pattern in greeting_patterns):
            # Define greeting responses
            greeting_responses = [
                "Hello! How can I help you with SUNY/CUNY information today?",
                "Hi there! What would you like to know about SUNY or CUNY schools?",
                "Hey! I'm here to assist with your questions about the SUNY or CUNY system. What can I help you with?",
                "Greetings! I'm your SUNY/CUNY assistant. How can I assist you today?",
                "Hello! I'm ready to help with your SUNY/CUNY questions. What would you like to know?"
            ]
            
            # Always use the first greeting for consistency
            state["response"] = greeting_responses[0]
            state["agent_type"] = "greeting"
            state["sources"] = None
            
            # Set metadata for greeting (no model used for this)
            state["system_metadata"]["model_used"] = "direct_response"
            state["system_metadata"]["provider"] = "system"
            
            # Return greeting response directly
            return QueryResponse(
                response=state["response"],
                agent_type=state["agent_type"],
                sources=state["sources"],
                original_query=state["original_query"],
                system_metadata=SystemMetadata(**state["system_metadata"])
            )
        
        # Handle off-topic detection first using the agent routing
        router_state = determine_agent_type(state)
        
        # If the router handled it directly (off-topic), return that result
        if isinstance(router_state, QueryResponse):
            return router_state
        
        # Process based on agent_type
        agent_type = router_state["agent_type"]
        logger.info(f"Direct processing for agent_type: {agent_type}")
        
        # Process the query with the appropriate agent
        result_state = None
        if agent_type == "browser":
            result_state = browser_agent(router_state)
        elif agent_type == "transfer":
            result_state = transfer_agent(router_state)
        elif agent_type == "recommendation":
            result_state = recommendation_agent(router_state)
        elif agent_type == "professor":
            result_state = professor_agent(router_state)
        else:
            # Fallback to browser agent if unknown agent type
            logger.warning(f"Unknown agent type '{agent_type}', falling back to browser agent")
            result_state = browser_agent(router_state)
            
        # Log the metadata to debug
        logger.info(f"Direct processing complete. Final metadata: {result_state.get('system_metadata')}")
        
        # Apply safety filter to response
        response_text = result_state.get("response", "No response generated")
        if safety_filter:
            filtered_response, was_modified = safety_filter.filter_response(response_text, request.query)
            if was_modified:
                logger.info(f"Response was modified by safety filter for query: {request.query}")
                result_state["response"] = filtered_response
                # Update metadata to show filtering was applied
                if "system_metadata" in result_state:
                    result_state["system_metadata"]["filtered_by_safety"] = True
        
        # Return the structured response with complete metadata
        metadata = result_state.get("system_metadata", {})
        logger.info(f"Model used: {metadata.get('model_used')}, Provider: {metadata.get('provider')}")
        
        return QueryResponse(
            response=result_state.get("response", "No response generated"),
            agent_type=result_state.get("agent_type", "unknown"),
            sources=result_state.get("sources"),
            original_query=result_state.get("original_query"),
            system_metadata=SystemMetadata(
                model_used=metadata.get("model_used", "unknown"),
                embedding_model=metadata.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
                provider=metadata.get("provider", "unknown"),
                timestamp=metadata.get("timestamp", datetime.datetime.utcnow().isoformat())
            )
        )
        
    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        # Get detailed exception info for debugging
        error_details = traceback.format_exc()
        logger.error(f"Detailed error: {error_details}")
        
        # Return error response
        return QueryResponse(
            response=f"I'm sorry, an error occurred while processing your query: {str(e)}",
            agent_type="error",
            sources=None,
            system_metadata=SystemMetadata(
                model_used="none",
                embedding_model="sentence-transformers/all-mpnet-base-v2",
                provider="system",
                timestamp=datetime.datetime.utcnow().isoformat()
            )
        )

# Add the new endpoint after the existing "query" endpoint
@app.post("/professor", response_model=QueryResponse)
async def professor_query(request: ProfessorRequest):
    """
    Specialized endpoint for professor-specific queries. 
    Takes professor's first and last name, college name, and a specific question about the professor.
    """
    try:
        # Construct a structured query for the professor agent
        formatted_query = f"What can you tell me about Professor {request.first_name} {request.last_name} at {request.college_name}? {request.question}"
        
        # Apply safety filter to incoming query
        if safety_filter:
            is_safe, explanation = safety_filter.check_query(formatted_query)
            if not is_safe:
                # Return a safe response explaining why the query was rejected
                return QueryResponse(
                    response="I'm sorry, but I cannot respond to that query as it may violate content safety policies. Please ask a different question about the professor that is related to academic matters.",
                    agent_type="safety_filtered",
                    sources=None,
                    original_query=formatted_query,
                    system_metadata=SystemMetadata(
                        model_used="safety_filter",
                        embedding_model="sentence-transformers/all-mpnet-base-v2", 
                        provider="system",
                        timestamp=datetime.datetime.utcnow().isoformat()
                    )
                )
        
        # Initialize state
        state = AgentState(query=formatted_query, user_id=request.user_id)
        
        # Clear any previous agent results
        state["response"] = None
        state["agent_type"] = "professor"  # Directly setting the agent type
        state["sources"] = None
        state["system_metadata"] = {
            "embedding_model": "sentence-transformers/all-mpnet-base-v2",
            "model_used": None,  # Will be populated during agent execution
            "provider": None,    # Will be populated during agent execution
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
        
        # Store original query and structured components for easier processing
        state["original_query"] = formatted_query
        state["professor_details"] = {
            "first_name": request.first_name,
            "last_name": request.last_name,
            "college_name": request.college_name,
            "specific_question": request.question
        }
        
        logger.info(f"Routing query directly to professor agent: {formatted_query}")
        
        # Process directly with the professor agent
        result_state = professor_agent(state)
            
        # Log the metadata for debugging
        logger.info(f"Professor endpoint processing complete. Final metadata: {result_state.get('system_metadata')}")
        
        # Apply safety filter to response
        response_text = result_state.get("response", "No response generated")
        if safety_filter:
            filtered_response, was_modified = safety_filter.filter_response(response_text, formatted_query)
            if was_modified:
                logger.info(f"Response was modified by safety filter for query: {formatted_query}")
                result_state["response"] = filtered_response
                # Update metadata to show filtering was applied
                if "system_metadata" in result_state:
                    result_state["system_metadata"]["filtered_by_safety"] = True
        
        # Return the structured response with complete metadata
        metadata = result_state.get("system_metadata", {})
        logger.info(f"Model used: {metadata.get('model_used')}, Provider: {metadata.get('provider')}")
        
        return QueryResponse(
            response=result_state.get("response", "No response generated"),
            agent_type=result_state.get("agent_type", "professor"),
            sources=result_state.get("sources"),
            original_query=result_state.get("original_query"),
            system_metadata=SystemMetadata(
                model_used=metadata.get("model_used", "unknown"),
                embedding_model=metadata.get("embedding_model", "sentence-transformers/all-mpnet-base-v2"),
                provider=metadata.get("provider", "unknown"),
                timestamp=metadata.get("timestamp", datetime.datetime.utcnow().isoformat())
            )
        )
        
    except Exception as e:
        logger.exception(f"Error processing professor query: {e}")
        # Get detailed exception info for debugging
        error_details = traceback.format_exc()
        logger.error(f"Detailed error: {error_details}")
        
        # Return error response
        return QueryResponse(
            response=f"I'm sorry, an error occurred while processing your professor query: {str(e)}",
            agent_type="error",
            sources=None,
            system_metadata=SystemMetadata(
                model_used="none",
                embedding_model="sentence-transformers/all-mpnet-base-v2",
                provider="system",
                timestamp=datetime.datetime.utcnow().isoformat()
            )
        )

# --- Uvicorn Runner ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    # Use host="0.0.0.0" to make it accessible on your network
    # Use reload=True for development (auto-restarts on code changes)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)