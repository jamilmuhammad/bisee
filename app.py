import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv

import pymongo
import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import groq
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('financial_insights.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/dbname")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "rag_chatbot")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    query_type: str

# Database connections
class DatabaseManager:
    def __init__(self):
        self.mongo_client = pymongo.MongoClient(Config.MONGODB_URL)
        self.mongo_db = self.mongo_client[Config.DATABASE_NAME]
        self.sessions_collection = self.mongo_db.sessions
        
    def get_postgres_connection(self):
        return psycopg2.connect(Config.POSTGRES_URL, cursor_factory=RealDictCursor)
    
    def save_session(self, session_id: str, message: str, response: str, query_type: str):
        session_data = {
            "session_id": session_id,
            "timestamp": datetime.utcnow(),
            "user_message": message,
            "bot_response": response,
            "query_type": query_type
        }
        self.sessions_collection.insert_one(session_data)
    
    def get_session_context(self, session_id: str, limit: int = 5) -> List[Dict]:
        return list(self.sessions_collection.find(
            {"session_id": session_id}
        ).sort("timestamp", -1).limit(limit))

# Database Schema Inspector
class SchemaInspector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._schema_cache = None
    
    def get_schema_info(self) -> Dict[str, Any]:
        if self._schema_cache:
            return self._schema_cache
            
        with self.db_manager.get_postgres_connection() as conn:
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            tables = [row['table_name'] for row in cursor.fetchall()]
            
            schema_info = {}
            for table in tables:
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns 
                    WHERE table_name = '{table}'
                    ORDER BY ordinal_position
                """)
                columns = cursor.fetchall()
                schema_info[table] = {
                    'columns': columns,
                    'sample_data': self._get_sample_data(cursor, table)
                }
        
        self._schema_cache = schema_info
        return schema_info
    
    def _get_sample_data(self, cursor, table_name: str) -> List[Dict]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            return cursor.fetchall()
        except:
            return []

# SQL Query Generator
class SQLQueryGenerator:
    def __init__(self, llm: ChatGroq, schema_info: Dict[str, Any]):
        self.llm = llm
        self.schema_info = schema_info
        
        self.query_template = PromptTemplate(
            input_variables=["user_input", "schema_info", "context"],
            template="""
You are a SQL query generator. Generate ONLY safe SELECT queries for PostgreSQL based on the user input and database schema.

STRICT LIMITATIONS:
- ONLY SELECT queries allowed (no INSERT, UPDATE, DELETE, DROP, ALTER, etc.)
- Only simple operations: COUNT, AVG, SUM, MIN, MAX, basic WHERE clauses
- No complex JOINs, subqueries, or window functions
- No user-defined functions or stored procedures

Database Schema:
{schema_info}

Previous Context:
{context}

User Input: {user_input}

Generate a safe SQL query that answers the user's question. If the request is not supported, respond with "UNSUPPORTED_QUERY".

SQL Query:
"""
        )
    
    def generate_query(self, user_input: str, context: str = "") -> str:
        schema_str = self._format_schema()
        
        prompt = self.query_template.format(
            user_input=user_input,
            schema_info=schema_str,
            context=context
        )
        
        response = self.llm.invoke(prompt)
        query = response.content.strip().split("\nSQL Query:")[-1].strip()
        match = re.search(r"```sql\n(.*?)\n```", query, re.DOTALL)
        sql_query = match.group(1).strip() if match else None

        logger.info(f"Generated SQL Query: {sql_query}")
        
        # Validate query safety
        if not self._is_safe_query(sql_query):
            return "UNSUPPORTED_QUERY"
            
        return sql_query if sql_query else "UNSUPPORTED_QUERY"
    
    def _format_schema(self) -> str:
        schema_str = ""
        for table, info in self.schema_info.items():
            schema_str += f"\nTable: {table}\n"
            schema_str += "Columns:\n"
            for col in info['columns']:
                schema_str += f"  - {col['column_name']} ({col['data_type']})\n"
            if info['sample_data']:
                schema_str += f"Sample data: {info['sample_data'][:2]}\n"
        return schema_str
    
    def _is_safe_query(self, query: str) -> bool:
        query_upper = query.upper().strip()
        
        # Must start with SELECT
        if not query_upper.startswith("SELECT"):
            return False
        
        # Blocked keywords
        blocked_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'UNION', 'INTERSECT', 'EXCEPT'  # Prevent complex queries
        ]
        
        for keyword in blocked_keywords:
            if keyword in query_upper:
                return False
        
        return True

# Query Executor
class QueryExecutor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        try:
            with self.db_manager.get_postgres_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description]
                
                return {
                    "success": True,
                    "data": results,
                    "columns": column_names,
                    "row_count": len(results)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "columns": [],
                "row_count": 0
            }

# LangGraph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    session_id: str
    query_type: str
    context: str
    sql_query: str
    query_results: Dict[str, Any]
    final_response: str

# Agent nodes
class SQLAgent:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.schema_inspector = SchemaInspector(self.db_manager)
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        self.query_generator = SQLQueryGenerator(
            self.llm, 
            self.schema_inspector.get_schema_info()
        )
        self.query_executor = QueryExecutor(self.db_manager)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("format_response", self.format_response)
        workflow.add_node("handle_general", self.handle_general)
        
        # Add edges
        workflow.set_entry_point("route_query")
        workflow.add_conditional_edges(
            "route_query",
            self.route_decision,
            {
                "sql_query": "generate_sql",
                "general": "handle_general"
            }
        )
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_response")
        workflow.add_edge("handle_general", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def route_query(self, state: AgentState) -> AgentState:
        user_input = state["user_input"]
        
        # Simple routing logic
        sql_keywords = ["count", "average", "sum", "total", "how many", "show me", "list", "find"]
        
        query_type = "sql_query" if any(keyword in user_input.lower() for keyword in sql_keywords) else "general"
        
        # Get session context
        context = ""
        if state["session_id"]:
            session_data = self.db_manager.get_session_context(state["session_id"])
            context = "\n".join([f"User: {s['user_message']}\nBot: {s['bot_response']}" for s in session_data])
        
        state["query_type"] = query_type
        state["context"] = context
        return state
    
    def route_decision(self, state: AgentState) -> str:
        return state["query_type"]
    
    def generate_sql(self, state: AgentState) -> AgentState:
        sql_query = self.query_generator.generate_query(
            state["user_input"], 
            state["context"]
        )
        state["sql_query"] = sql_query
        return state
    
    def execute_sql(self, state: AgentState) -> AgentState:
        if state["sql_query"] == "UNSUPPORTED_QUERY":
            state["query_results"] = {
                "success": False,
                "error": "Query not supported. Only simple SELECT queries are allowed.",
                "data": [],
                "columns": [],
                "row_count": 0
            }
        else:
            state["query_results"] = self.query_executor.execute_query(state["sql_query"])
        return state
    
    def handle_general(self, state: AgentState) -> AgentState:
        # For general queries, provide helpful information about available data
        response = f"""I can help you query your database. I support simple queries like:
- Counting records: "How many users are there?"
- Averages: "What's the average order value?"
- Sums: "What's the total revenue?"
- Finding records: "Show me recent orders"

Available tables in your database:
{', '.join(self.schema_inspector.get_schema_info().keys())}

What would you like to know about your data?"""
        
        state["final_response"] = response
        return state
    
    def format_response(self, state: AgentState) -> AgentState:
        if state["query_type"] == "general":
            return state
        
        results = state["query_results"]
        
        if not results["success"]:
            state["final_response"] = f"Sorry, I couldn't process your query: {results['error']}"
        else:
            if results["row_count"] == 0:
                state["final_response"] = "No data found matching your query."
            else:
                # Format results in a readable way
                data = results["data"]
                columns = results["columns"]
                
                if len(data) == 1 and len(columns) == 1:
                    # Single value result (like COUNT, AVG)
                    state["final_response"] = f"Result: {data[0][columns[0]]}"
                else:
                    # Multiple rows/columns
                    response = f"Found {results['row_count']} result(s):\n\n"
                    for row in data[:10]:  # Limit to 10 rows
                        row_str = []
                        for col in columns:
                            row_str.append(f"{col}: {row[col]}")
                        response += "- " + ", ".join(row_str) + "\n"
                    
                    if results["row_count"] > 10:
                        response += f"\n... and {results['row_count'] - 10} more rows"
                    
                    state["final_response"] = response
        
        return state
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if not session_id:
            session_id = str(uuid.uuid4())
        
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            user_input=message,
            session_id=session_id,
            query_type="",
            context="",
            sql_query="",
            query_results={},
            final_response=""
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Save to session
        self.db_manager.save_session(
            session_id,
            message,
            final_state["final_response"],
            final_state["query_type"]
        )
        
        return {
            "response": final_state["final_response"],
            "session_id": session_id,
            "query_type": final_state["query_type"]
        }

# FastAPI app
app = FastAPI(title="RAG SQL Agent Chatbot", version="1.0.0")
agent = SQLAgent()

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        result = agent.process_message(request.message, request.session_id)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/schema")
async def get_schema():
    return agent.schema_inspector.get_schema_info()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)