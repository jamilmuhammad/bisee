import re
import uuid
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated

from app.core.config import settings
from app.services.database_service import DatabaseManager
from app.models.rag_models import ChatRequest, ChatResponse, AnalysisState

logger = logging.getLogger(__name__)


# LangGraph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    session_id: str
    user_id: Optional[str]
    user_profile: Optional[Dict[str, Any]]
    query_types: List[str]
    context: str
    sql_query: str
    query_results: Dict[str, Any]
    query_data: List[Dict[str, Any]]
    final_response: str
    reflection_feedback: Optional[str]
    refined_response: Optional[str]
    analysis_type: Optional[str]
    forecast_results: Optional[Dict[str, Any]]
    simulation_results: Optional[Dict[str, Any]]
    predictive_model: Optional[str]
    simulation_parameters: Optional[Dict[str, Any]]
    data_validation: Optional[Dict[str, Any]]


class QueryExecutor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        try:
            with self.db_manager.get_postgres_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                
                # Fetch results
                results = cursor.fetchall()
                
                return {
                    "success": True,
                    "data": [dict(row) for row in results],
                    "columns": columns,
                    "row_count": len(results),
                    "error": None
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "data": [],
                "columns": [],
                "row_count": 0
            }


class SchemaInspector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._schema_cache = None
    
    def get_schema_info(self) -> Dict[str, Any]:
        if self._schema_cache:
            return self._schema_cache
            
        try:
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
                    # Get columns for each table
                    cursor.execute(f"""
                        SELECT column_name, data_type, is_nullable
                        FROM information_schema.columns 
                        WHERE table_name = '{table}'
                        ORDER BY ordinal_position
                    """)
                    columns = cursor.fetchall()
                    
                    schema_info[table] = {
                        'columns': [
                            {
                                'name': col['column_name'],
                                'type': col['data_type'],
                                'nullable': col['is_nullable'] == 'YES'
                            }
                            for col in columns
                        ],
                        'sample_data': self._get_sample_data(cursor, table)
                    }
            
            self._schema_cache = schema_info
            return schema_info
        except Exception as e:
            logger.error(f"Failed to get schema info: {e}")
            return {}
    
    def _get_sample_data(self, cursor, table_name: str) -> List[Dict]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            return [dict(row) for row in cursor.fetchall()]
        except:
            return []


class PromptTemplates:
    @staticmethod
    def get_router_prompt():
        return PromptTemplate(
            input_variables=["user_input", "context"],
            template="""
You are a query router that classifies user inputs into different categories.

Categories:
1. "sql_query" - Questions about data analysis, counts, statistics, reports, or database queries
2. "visualization" - Requests for charts, graphs, plots, or visual representations
3. "general" - General questions, greetings, or unclear requests

Context from previous conversation:
{context}

User Input: {user_input}

Based on the user input, classify this into one of the three categories above.
Look for keywords like:
- SQL: "count", "how many", "show", "list", "average", "sum", "total", "find", "data"
- Visualization: "chart", "graph", "plot", "visualize", "show chart", "bar chart", "pie chart"
- General: greetings, unclear requests, non-data related questions

Return only the category name: sql_query, visualization, or general
"""
        )

    @staticmethod
    def get_sql_generator_prompt():
        return PromptTemplate(
            input_variables=["user_input", "schema_info", "context"],
            template="""
You are an expert SQL query generator for PostgreSQL. Generate safe, efficient SELECT queries only.

STRICT RULES:
- ONLY SELECT queries (no INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, etc.)
- Use proper PostgreSQL syntax
- Include appropriate WHERE clauses, ORDER BY, LIMIT as needed
- Use aggregate functions (COUNT, SUM, AVG, MIN, MAX) when appropriate
- No complex subqueries or CTEs unless absolutely necessary
- No user-defined functions or stored procedures

Database Schema:
{schema_info}

Previous Context:
{context}

User Request: {user_input}

Generate a clean, executable SQL query that answers the user's question.
If the request cannot be fulfilled safely, return "UNSUPPORTED_QUERY".

Return only the SQL query without any formatting or explanation.
"""
        )

    @staticmethod
    def get_general_response_prompt():
        return PromptTemplate(
            input_variables=["user_input", "available_tables", "context"],
            template="""
You are a helpful database assistant that provides general information and guidance.

User Input: {user_input}
Available Tables: {available_tables}
Context: {context}

Provide a helpful response that:
1. Addresses the user's question or comment
2. Offers guidance on what they can do with the available data
3. Suggests example queries they might find useful
4. Maintains a friendly, professional tone

If the user is greeting you, respond appropriately and explain your capabilities.
"""
        )


class SQLQueryGenerator:
    def __init__(self, llm: ChatGroq, schema_info: Dict[str, Any]):
        self.llm = llm
        self.schema_info = schema_info
        self.query_prompt = PromptTemplates.get_sql_generator_prompt()
    
    def generate_query(self, user_input: str, context: str = "") -> str:
        schema_str = self._format_schema()
        
        prompt = self.query_prompt.format(
            user_input=user_input,
            schema_info=schema_str,
            context=context
        )
        
        response = self.llm.invoke(prompt)
        query = response.content.strip()
        
        # Clean up the query
        query = self._clean_query(query)
        
        logger.info(f"Generated SQL Query: {query}")
        
        # Validate query safety
        if not self._is_safe_query(query):
            return "UNSUPPORTED_QUERY"
            
        return query if query else "UNSUPPORTED_QUERY"
    
    def _clean_query(self, query: str) -> str:
        """Clean and format the SQL query"""
        # Remove markdown formatting
        query = re.sub(r'```sql\n(.*?)\n```', r'\1', query, flags=re.DOTALL)
        query = re.sub(r'```\n(.*?)\n```', r'\1', query, flags=re.DOTALL)
        
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        return query.strip()
    
    def _format_schema(self) -> str:
        schema_str = ""
        for table, info in self.schema_info.items():
            schema_str += f"\nTable: {table}\n"
            schema_str += "Columns:\n"
            for col in info['columns']:
                schema_str += f"  - {col['name']} ({col['type']})\n"
            
            if info['sample_data']:
                schema_str += f"Sample data: {info['sample_data'][:2]}\n"

        return schema_str
    
    def _is_safe_query(self, query: str) -> bool:
        query_upper = query.upper().strip()

        # Must start with SELECT
        if not query_upper.startswith("SELECT"):
            return "UNSUPPORTED_QUERY"

        # Blocked keywords
        blocked_keywords = [
            'INSERT', 'UPDATE', 'DELETE', 'DROP', 'ALTER', 'CREATE',
            'TRUNCATE', 'GRANT', 'REVOKE', 'EXEC', 'EXECUTE',
            'UNION', 'INTERSECT', 'EXCEPT'
        ]
        
        for keyword in blocked_keywords:
            if keyword in query_upper:
                return "UNSUPPORTED_QUERY"

        return True


class RAGChatbotService:
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.query_executor = QueryExecutor(self.db_manager)
        self.schema_inspector = SchemaInspector(self.db_manager)
        self.llm = ChatGroq(
            temperature=0,
            model_name=settings.LLM_MODEL_NAME_GROQ,
            groq_api_key=settings.GROQ_API_KEY
        )
        self.sql_generator = SQLQueryGenerator(self.llm, self.schema_inspector.get_schema_info())
        self.router_prompt = PromptTemplates.get_router_prompt()
        self.general_prompt = PromptTemplates.get_general_response_prompt()
        
        # Initialize LangGraph workflow
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("router", self._router_node)
        workflow.add_node("sql_generator", self._sql_generator_node)
        workflow.add_node("query_executor", self._query_executor_node)
        workflow.add_node("response_generator", self._response_generator_node)
        workflow.add_node("general_responder", self._general_responder_node)
        
        # Add edges
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "sql_query": "sql_generator",
                "general": "general_responder",
                "visualization": "general_responder"  # For now, treat as general
            }
        )
        workflow.add_edge("sql_generator", "query_executor")
        workflow.add_edge("query_executor", "response_generator")
        workflow.add_edge("response_generator", END)
        workflow.add_edge("general_responder", END)
        
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Route the user input to appropriate handler"""
        prompt = self.router_prompt.format(
            user_input=state["user_input"],
            context=state["context"]
        )
        
        response = self.llm.invoke(prompt)
        query_type = response.content.strip().lower()
        
        state["query_types"] = [query_type]
        state["messages"].append(HumanMessage(content=state["user_input"]))
        
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Decision function for routing"""
        query_type = state["query_types"][0] if state["query_types"] else "general"
        return query_type
    
    def _sql_generator_node(self, state: AgentState) -> AgentState:
        """Generate SQL query"""
        sql_query = self.sql_generator.generate_query(
            state["user_input"],
            state["context"]
        )
        
        state["sql_query"] = sql_query
        return state
    
    def _query_executor_node(self, state: AgentState) -> AgentState:
        """Execute the SQL query"""
        if state["sql_query"] == "UNSUPPORTED_QUERY":
            state["query_results"] = {
                "success": False,
                "error": "Query type not supported or unsafe",
                "data": [],
                "columns": [],
                "row_count": 0
            }
        else:
            state["query_results"] = self.query_executor.execute_query(state["sql_query"])
        
        state["query_data"] = state["query_results"].get("data", [])
        return state
    
    def _response_generator_node(self, state: AgentState) -> AgentState:
        """Generate response from query results"""
        if state["query_results"]["success"]:
            response = self._format_query_response(
                state["user_input"],
                state["query_results"],
                state["sql_query"]
            )
        else:
            response = f"I encountered an error while processing your request: {state['query_results']['error']}"
        
        state["final_response"] = response
        state["messages"].append(AIMessage(content=response))
        
        return state
    
    def _general_responder_node(self, state: AgentState) -> AgentState:
        """Handle general queries"""
        available_tables = list(self.schema_inspector.get_schema_info().keys())
        
        prompt = self.general_prompt.format(
            user_input=state["user_input"],
            available_tables=available_tables,
            context=state["context"]
        )
        
        response = self.llm.invoke(prompt)
        state["final_response"] = response.content.strip()
        state["messages"].append(AIMessage(content=state["final_response"]))
        
        return state
    
    def _format_query_response(self, user_input: str, query_results: Dict[str, Any], sql_query: str) -> str:
        """Format the query results into a readable response"""
        if not query_results["success"]:
            return f"I encountered an error: {query_results['error']}"
        
        data = query_results["data"]
        row_count = query_results["row_count"]
        
        if row_count == 0:
            return "No data found matching your criteria."
        
        # Create a formatted response
        response = f"I found {row_count} result(s) for your query:\n\n"
        
        if row_count <= 10:
            # Show all results if 10 or fewer
            for i, row in enumerate(data, 1):
                response += f"{i}. "
                response += ", ".join([f"{k}: {v}" for k, v in row.items()])
                response += "\n"
        else:
            # Show first 5 results and summary
            for i, row in enumerate(data[:5], 1):
                response += f"{i}. "
                response += ", ".join([f"{k}: {v}" for k, v in row.items()])
                response += "\n"
            response += f"\n... and {row_count - 5} more results."
        
        return response
    
    async def process_chat(self, request: ChatRequest, user_id: Optional[str] = None) -> ChatResponse:
        """Process a chat request"""
        try:
            # Generate session ID if not provided
            session_id = request.session_id or str(uuid.uuid4())
            
            # Get session context
            context_history = self.db_manager.get_session_context(session_id, user_id)
            context = self._format_context(context_history)
            
            # Initialize state
            initial_state = {
                "messages": [],
                "user_input": request.message,
                "session_id": session_id,
                "user_id": user_id,
                "user_profile": None,
                "query_types": [],
                "context": context,
                "sql_query": "",
                "query_results": {},
                "query_data": [],
                "final_response": "",
                "reflection_feedback": None,
                "refined_response": None,
                "analysis_type": None,
                "forecast_results": None,
                "simulation_results": None,
                "predictive_model": None,
                "simulation_parameters": None,
                "data_validation": None
            }
            
            # Run the workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Save session
            query_type = final_state["query_types"][0] if final_state["query_types"] else "general"
            self.db_manager.save_session(
                session_id=session_id,
                message=request.message,
                response=final_state["final_response"],
                query_type=query_type,
                user_id=user_id,
                sql_query=final_state.get("sql_query"),
                query_results=final_state.get("query_results")
            )
            
            # Create response
            return ChatResponse(
                response=final_state["final_response"],
                session_id=session_id,
                query_types=final_state["query_types"],
                context=context,
                sql_query=final_state.get("sql_query"),
                query_results=final_state.get("query_results"),
                query_data=final_state.get("query_data"),
                final_response=final_state["final_response"],
                reflection_feedback=final_state.get("reflection_feedback"),
                refined_response=final_state.get("refined_response"),
                analysis_state=AnalysisState(
                    analysis_type=final_state.get("analysis_type"),
                    forecast_results=final_state.get("forecast_results"),
                    simulation_results=final_state.get("simulation_results"),
                    predictive_model=final_state.get("predictive_model"),
                    simulation_parameters=final_state.get("simulation_parameters"),
                    data_validation=final_state.get("data_validation")
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing chat: {e}")
            return ChatResponse(
                response=f"I encountered an error while processing your request: {str(e)}",
                session_id=request.session_id or str(uuid.uuid4()),
                query_types=["error"],
                context="",
                sql_query=None,
                query_results=None,
                query_data=None,
                final_response=f"Error: {str(e)}",
                reflection_feedback=None,
                refined_response=None,
                analysis_state=None
            )
    
    def _format_context(self, context_history: List[Dict]) -> str:
        """Format context history for the prompt"""
        if not context_history:
            return "No previous conversation."
        
        context = "Previous conversation:\n"
        for session in reversed(context_history[-3:]):  # Last 3 interactions
            context += f"User: {session.get('user_message', 'N/A')}\n"
            context += f"Assistant: {session.get('bot_response', 'N/A')}\n\n"
        
        return context
    
    def get_user_chat_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's chat history"""
        return self.db_manager.get_user_chat_history(user_id, limit)
