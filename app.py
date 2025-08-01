import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from dotenv import load_dotenv
import json
import base64
from io import BytesIO

import pymongo
import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
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

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Predictive analysis imports
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Time series forecasting imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

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
    POSTGRES_URL: Optional[str] = os.getenv("POSTGRES_URL")
    DATABASE_NAME = os.getenv("DATABASE_NAME", "rag_chatbot")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Pydantic models for structured responses
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    query_type: str
    generated_sql: Optional[str] = None
    data_query_result: Optional[Dict[str, Any]] = None
    markdown_result: Optional[str] = None
    visualization_result: Optional[str] = None  # Base64 encoded image

class VisualizationRequest(BaseModel):
    chart_type: str
    data: List[Dict[str, Any]]
    columns: List[str]
    title: str
    x_axis: Optional[str] = None
    y_axis: Optional[str] = None

# Database connections
class DatabaseManager:
    def __init__(self):
        self.mongo_client = pymongo.MongoClient(Config.MONGODB_URL)
        self.mongo_db = self.mongo_client[Config.DATABASE_NAME]
        self.sessions_collection = self.mongo_db.sessions
        
    def get_postgres_connection(self):
        if not Config.POSTGRES_URL:
            raise ValueError("PostgreSQL URL is not configured.")
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

# Database Schema Inspector
class SchemaInspector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._schema_cache = None
    
    @st.cache_data(ttl=600)
    def get_schema_info(_self) -> Dict[str, Any]:
        if _self._schema_cache:
            return _self._schema_cache
            
        with _self.db_manager.get_postgres_connection() as conn:
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
                    'sample_data': _self._get_sample_data(cursor, table)
                }
        
        _self._schema_cache = schema_info
        return schema_info
    
    def _get_sample_data(self, cursor, table_name: str) -> List[Dict]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            return cursor.fetchall()
        except:
            return []

# Prompt Templates for different agents
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
    def get_reflection_prompt():
        return PromptTemplate(
            input_variables=["user_input", "sql_query", "query_results", "current_response"],
            template="""
You are a reflection agent that reviews and improves responses for accuracy and completeness.

User Question: {user_input}
Generated SQL: {sql_query}
Query Results: {query_results}
Current Response: {current_response}

Review the current response and provide feedback on:
1. Accuracy - Does it correctly answer the user's question?
2. Completeness - Are all aspects of the question addressed?
3. Clarity - Is the response clear and understandable?
4. Data interpretation - Are the results properly interpreted?

Provide specific feedback and suggestions for improvement.
If the response is good, simply return "APPROVED".
"""
        )

    @staticmethod
    def get_response_refiner_prompt():
        return PromptTemplate(
            input_variables=["user_input", "sql_query", "query_results", "current_response", "reflection_feedback"],
            template="""
You are a response refinement agent that improves responses based on reflection feedback.

User Question: {user_input}
Generated SQL: {sql_query}
Query Results: {query_results}
Current Response: {current_response}
Reflection Feedback: {reflection_feedback}

Based on the reflection feedback, create an improved, more accurate and precise response.
Make sure to:
1. Address all points raised in the feedback
2. Provide clear, actionable insights
3. Use proper formatting and structure
4. Include relevant context and explanations

Return the refined response:
"""
        )

    @staticmethod
    def get_visualization_prompt():
        return PromptTemplate(
            input_variables=["user_input", "query_results", "columns"],
            template="""
You are a visualization recommendation agent that suggests appropriate chart types based on data.

User Request: {user_input}
Query Results: {query_results}
Available Columns: {columns}

Based on the data and user request, recommend the best visualization type and configuration.

Available chart types:
- bar: For categorical data comparison
- line: For time series or trend data
- pie: For proportional data (max 10 categories)
- scatter: For correlation between two variables
- histogram: For distribution of numerical data

Return a JSON object with:
{{
    "chart_type": "bar|line|pie|scatter|histogram",
    "title": "Chart title",
    "x_axis": "column name for x-axis",
    "y_axis": "column name for y-axis",
    "description": "Brief description of what the chart shows"
}}
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

# Enhanced LangGraph State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    user_input: str
    session_id: str
    query_types: List[str]  # Changed from query_type to query_types array
    context: str
    sql_query: str
    query_results: Dict[str, Any]
    query_data: List[Dict[str, Any]]  # New variable for only query result data
    final_response: str
    # New fields for enhanced response
    generated_sql: Optional[str]
    data_query_result: Optional[Dict[str, Any]]
    markdown_result: Optional[str]
    visualization_result: Optional[str]
    reflection_feedback: Optional[str]
    refined_response: Optional[str]
    chart_type: Optional[str]
    visualization_data: Optional[Dict[str, Any]]
    # New fields for predictive and prescriptive analysis
    analysis_type: Optional[str]  # descriptive, predictive, prescriptive
    forecast_results: Optional[Dict[str, Any]]
    simulation_results: Optional[Dict[str, Any]]
    predictive_model: Optional[str]  # ARIMA, Prophet, Linear, etc.
    simulation_parameters: Optional[Dict[str, Any]]

# Database Schema Inspector
class SchemaInspector:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self._schema_cache = None
    
    @st.cache_data(ttl=600)
    def get_schema_info(_self) -> Dict[str, Any]:
        if _self._schema_cache:
            return _self._schema_cache
            
        with _self.db_manager.get_postgres_connection() as conn:
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
                    'sample_data': _self._get_sample_data(cursor, table)
                }
        
        _self._schema_cache = schema_info
        return schema_info
    
    def _get_sample_data(self, cursor, table_name: str) -> List[Dict]:
        try:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            return cursor.fetchall()
        except:
            return []

# Prompt Templates for different agents
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
    def get_reflection_prompt():
        return PromptTemplate(
            input_variables=["user_input", "sql_query", "query_results", "current_response"],
            template="""
You are a reflection agent that reviews and improves responses for accuracy and completeness.

User Question: {user_input}
Generated SQL: {sql_query}
Query Results: {query_results}
Current Response: {current_response}

Review the current response and provide feedback on:
1. Accuracy - Does it correctly answer the user's question?
2. Completeness - Are all aspects of the question addressed?
3. Clarity - Is the response clear and understandable?
4. Data interpretation - Are the results properly interpreted?

Provide specific feedback and suggestions for improvement.
If the response is good, simply return "APPROVED".
"""
        )

    @staticmethod
    def get_response_refiner_prompt():
        return PromptTemplate(
            input_variables=["user_input", "sql_query", "query_results", "current_response", "reflection_feedback"],
            template="""
You are a response refinement agent that improves responses based on reflection feedback.

User Question: {user_input}
Generated SQL: {sql_query}
Query Results: {query_results}
Current Response: {current_response}
Reflection Feedback: {reflection_feedback}

Based on the reflection feedback, create an improved, more accurate and precise response.
Make sure to:
1. Address all points raised in the feedback
2. Provide clear, actionable insights
3. Use proper formatting and structure
4. Include relevant context and explanations

Return the refined response:
"""
        )

    @staticmethod
    def get_visualization_prompt():
        return PromptTemplate(
            input_variables=["user_input", "query_results", "columns"],
            template="""
You are a visualization recommendation agent that suggests appropriate chart types based on data.

User Request: {user_input}
Query Results: {query_results}
Available Columns: {columns}

Based on the data and user request, recommend the best visualization type and configuration.

Available chart types:
- bar: For categorical data comparison
- line: For time series or trend data
- pie: For proportional data (max 10 categories)
- scatter: For correlation between two variables
- histogram: For distribution of numerical data

Return a JSON object with:
{{
    "chart_type": "bar|line|pie|scatter|histogram",
    "title": "Chart title",
    "x_axis": "column name for x-axis",
    "y_axis": "column name for y-axis",
    "description": "Brief description of what the chart shows"
}}
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

    @staticmethod
    def get_predictive_analysis_prompt():
        return PromptTemplate(
            input_variables=["user_input", "query_data", "columns", "analysis_context"],
            template="""
You are a predictive analytics expert that analyzes historical data to make forecasts.

User Request: {user_input}
Historical Data: {query_data}
Available Columns: {columns}
Context: {analysis_context}

Based on the historical data provided, determine the best forecasting approach:

1. Identify time-based patterns (trends, seasonality)
2. Recommend appropriate forecasting model (ARIMA, Prophet, Linear Regression, etc.)
3. Suggest forecast horizon (periods to predict ahead)
4. Identify key variables for prediction

Return a JSON object with:
{{
    "model_type": "arima|prophet|linear|exponential_smoothing",
    "forecast_periods": "number of periods to forecast",
    "time_column": "column name containing dates/time",
    "target_column": "column name to forecast",
    "seasonality": "detected seasonality pattern if any",
    "trend": "detected trend pattern",
    "confidence_level": "0.95",
    "description": "Brief explanation of the forecasting approach"
}}
"""
        )

    @staticmethod
    def get_prescriptive_analysis_prompt():
        return PromptTemplate(
            input_variables=["user_input", "query_data", "columns", "descriptive_context"],
            template="""
You are a prescriptive analytics expert that performs what-if analysis and simulations.

User Request: {user_input}
Current Data: {query_data}
Available Columns: {columns}
Descriptive Context: {descriptive_context}

Based on the current data and user request, design a simulation or what-if analysis:

1. Identify variables to modify (independent variables)
2. Determine the target outcome (dependent variable)
3. Suggest simulation parameters and ranges
4. Recommend simulation scenarios

Return a JSON object with:
{{
    "simulation_type": "what_if|sensitivity|scenario|optimization",
    "independent_variables": ["list of columns to vary"],
    "dependent_variable": "target outcome column",
    "scenarios": [
        {{
            "name": "scenario name",
            "parameters": {{"column": "new_value"}},
            "description": "what this scenario tests"
        }}
    ],
    "parameter_ranges": {{"column": {{"min": value, "max": value, "step": value}}}},
    "description": "Brief explanation of the simulation approach"
}}
"""
        )

# SQL Query Generator
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
                schema_str += f"  - {col['column_name']} ({col['data_type']})\n"
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
            'UNION', 'INTERSECT', 'EXCEPT'  # Prevent complex queries
        ]
        
        for keyword in blocked_keywords:
            if keyword in query_upper:
                return "UNSUPPORTED_QUERY"

        return True

# Reflection Agent
class ReflectionAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.reflection_prompt = PromptTemplates.get_reflection_prompt()
        self.refiner_prompt = PromptTemplates.get_response_refiner_prompt()
    
    def reflect_on_response(self, user_input: str, sql_query: str, query_results: Dict[str, Any], current_response: str) -> str:
        """Reflect on the current response and provide feedback"""
        prompt = self.reflection_prompt.format(
            user_input=user_input,
            sql_query=sql_query,
            query_results=str(query_results),
            current_response=current_response
        )
        
        response = self.llm.invoke(prompt)
        return response.content.strip()
    
    def refine_response(self, user_input: str, sql_query: str, query_results: Dict[str, Any], 
                       current_response: str, reflection_feedback: str) -> str:
        """Refine the response based on reflection feedback"""
        if reflection_feedback.strip() == "APPROVED":
            return current_response
        
        prompt = self.refiner_prompt.format(
            user_input=user_input,
            sql_query=sql_query,
            query_results=str(query_results),
            current_response=current_response,
            reflection_feedback=reflection_feedback
        )
        
        response = self.llm.invoke(prompt)
        return response.content.strip()

# Visualization Agent
class VisualizationAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.viz_prompt = PromptTemplates.get_visualization_prompt()
        
    def generate_chart_config(self, user_input: str, query_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart configuration based on query results"""
        if not query_results.get("success") or not query_results.get("data"):
            return {"error": "No data available for visualization"}
        
        data = query_results["data"]
        columns = query_results["columns"]
        logger.info(data, "Query results data")
        logger.info(columns, "Query results columns")
        
        # Create a fallback configuration first
        fallback_config = {
            "chart_type": "bar",
            "title": "Data Visualization",
            "x_axis": columns[0] if columns else None,
            "y_axis": columns[1] if len(columns) > 1 else columns[0] if columns else None,
            "description": "Auto-generated chart from query results"
        }
        
        prompt = self.viz_prompt.format(
            user_input=user_input,
            query_results=str(data[:5]),  # First 5 rows for analysis
            columns=str(columns)
        )
        
        try:
            response = self.llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Check if response is empty
            if not response_content:
                logger.warning("Empty response from LLM for chart config, using fallback")
                return fallback_config
            
            # Try to extract JSON from response if it's wrapped in markdown
            if "```json" in response_content:
                # Extract JSON from markdown code block
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            elif "```" in response_content:
                # Extract content from any code block
                import re
                json_match = re.search(r'```\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            # Parse JSON
            config = json.loads(response_content)
            
            # Validate required fields and add defaults if missing
            if not isinstance(config, dict):
                logger.warning("Invalid config format from LLM, using fallback")
                return fallback_config
            
            # Ensure required fields exist
            config.setdefault("chart_type", fallback_config["chart_type"])
            config.setdefault("title", fallback_config["title"])
            config.setdefault("x_axis", fallback_config["x_axis"])
            config.setdefault("y_axis", fallback_config["y_axis"])
            config.setdefault("description", fallback_config["description"])
            
            return config
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in chart config: {e}. Response was: {response_content[:200]}...")
            return fallback_config
        except Exception as e:
            logger.error(f"Error generating chart config: {e}")
            return fallback_config
    
    def create_visualization(self, query_results: Dict[str, Any], chart_config: Dict[str, Any]) -> str:
        """Create visualization and return base64 encoded image"""
        try:
            if not query_results.get("success") or not query_results.get("data"):
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(query_results["data"])
            
            # Enhanced logic for axis determination based on your requirements
            # If data has 2+ rows and 2+ columns, use column 1 as labels for axis and the rest as data
            x_col = None
            y_col = None
            
            if len(df) >= 2 and len(df.columns) >= 2:
                # Column 1 (index 0) is label for the axis
                x_col = df.columns[0]
                # Column 2 (index 1) is the data for y-axis
                y_col = df.columns[1] if len(df.columns) > 1 else None
            
            # Override with chart config if provided
            chart_x_col = chart_config.get("x_axis")
            chart_y_col = chart_config.get("y_axis")
            
            if chart_x_col and chart_x_col in df.columns:
                x_col = chart_x_col
            if chart_y_col and chart_y_col in df.columns:
                y_col = chart_y_col
            
            # Create figure
            plt.figure(figsize=(10, 6))
            plt.style.use('seaborn-v0_8')
            
            chart_type = chart_config.get("chart_type", "bar")
            title = chart_config.get("title", "Data Visualization")
            
            if chart_type == "bar":
                self._create_bar_chart(df, x_col, y_col, title)
            elif chart_type == "line":
                self._create_line_chart(df, x_col, y_col, title)
            elif chart_type == "pie":
                self._create_pie_chart(df, x_col, y_col, title)
            elif chart_type == "scatter":
                self._create_scatter_chart(df, x_col, y_col, title)
            elif chart_type == "histogram":
                self._create_histogram(df, x_col, title)
            else:
                self._create_bar_chart(df, x_col, y_col, title)
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return None
    
    def _create_bar_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            plt.bar(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        else:
            # Auto-select columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.bar(range(len(df)), df[numeric_cols[0]])
                plt.ylabel(numeric_cols[0])
        plt.title(title)
        plt.xticks(rotation=45)
    
    def _create_line_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            plt.plot(df[x_col], df[y_col], marker='o')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.plot(df[numeric_cols[0]], marker='o')
                plt.ylabel(numeric_cols[0])
        plt.title(title)
        plt.xticks(rotation=45)
    
    def _create_pie_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            plt.pie(df[y_col], labels=df[x_col], autopct='%1.1f%%')
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.pie(df[numeric_cols[0]], autopct='%1.1f%%')
        plt.title(title)
    
    def _create_scatter_chart(self, df: pd.DataFrame, x_col: str, y_col: str, title: str):
        if x_col and y_col and x_col in df.columns and y_col in df.columns:
            plt.scatter(df[x_col], df[y_col])
            plt.xlabel(x_col)
            plt.ylabel(y_col)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                plt.scatter(df[numeric_cols[0]], df[numeric_cols[1]])
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
        plt.title(title)
    
    def _create_histogram(self, df: pd.DataFrame, x_col: str, title: str):
        if x_col and x_col in df.columns:
            plt.hist(df[x_col], bins=20)
            plt.xlabel(x_col)
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                plt.hist(df[numeric_cols[0]], bins=20)
                plt.xlabel(numeric_cols[0])
        plt.title(title)
        plt.ylabel('Frequency')

# Predictive Analysis Agent
class PredictiveAnalysisAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.predictive_prompt = PromptTemplates.get_predictive_analysis_prompt()
    
    def generate_forecast_config(self, user_input: str, query_data: List[Dict], columns: List[str], context: str = "") -> Dict[str, Any]:
        """Generate forecasting configuration based on historical data"""
        if not query_data or not columns:
            return {"error": "No data available for predictive analysis"}
        
        # Fallback configuration
        fallback_config = {
            "model_type": "linear",
            "forecast_periods": 12,
            "time_column": None,
            "target_column": columns[0] if columns else None,
            "confidence_level": 0.95,
            "description": "Simple linear forecast based on available data"
        }
        
        try:
            prompt = self.predictive_prompt.format(
                user_input=user_input,
                query_data=str(query_data[:10]),  # First 10 rows for analysis
                columns=str(columns),
                analysis_context=context
            )
            
            response = self.llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Parse JSON response with fallback
            if "```json" in response_content:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            config = json.loads(response_content)
            
            # Validate and set defaults
            config.setdefault("model_type", fallback_config["model_type"])
            config.setdefault("forecast_periods", fallback_config["forecast_periods"])
            config.setdefault("confidence_level", fallback_config["confidence_level"])
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating forecast config: {e}")
            return fallback_config
    
    def create_forecast(self, query_data: List[Dict], forecast_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create forecast based on configuration"""
        try:
            if not query_data:
                return {"error": "No data for forecasting"}
            
            df = pd.DataFrame(query_data)
            model_type = forecast_config.get("model_type", "linear")
            target_col = forecast_config.get("target_column")
            time_col = forecast_config.get("time_column")
            periods = int(forecast_config.get("forecast_periods", 12))
            
            if target_col not in df.columns:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                target_col = numeric_cols[0] if len(numeric_cols) > 0 else df.columns[0]
            
            if model_type == "arima" and STATSMODELS_AVAILABLE:
                return self._create_arima_forecast(df, target_col, periods)
            elif model_type == "prophet" and PROPHET_AVAILABLE:
                return self._create_prophet_forecast(df, target_col, time_col, periods)
            elif model_type == "exponential_smoothing" and STATSMODELS_AVAILABLE:
                return self._create_exponential_smoothing_forecast(df, target_col, periods)
            else:
                return self._create_linear_forecast(df, target_col, periods)
                
        except Exception as e:
            logger.error(f"Error creating forecast: {e}")
            return {"error": f"Forecasting failed: {str(e)}"}
    
    def _create_linear_forecast(self, df: pd.DataFrame, target_col: str, periods: int) -> Dict[str, Any]:
        """Create linear regression forecast"""
        try:
            values = df[target_col].values
            X = np.arange(len(values)).reshape(-1, 1)
            y = values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Generate forecasts
            future_X = np.arange(len(values), len(values) + periods).reshape(-1, 1)
            forecasts = model.predict(future_X)
            
            # Calculate basic confidence intervals (simplified)
            residuals = y - model.predict(X)
            mse = np.mean(residuals**2)
            std_error = np.sqrt(mse)
            confidence_interval = 1.96 * std_error  # 95% CI
            
            return {
                "model": "Linear Regression",
                "forecasts": forecasts.tolist(),
                "confidence_lower": (forecasts - confidence_interval).tolist(),
                "confidence_upper": (forecasts + confidence_interval).tolist(),
                "historical_values": values.tolist(),
                "mae": mean_absolute_error(y, model.predict(X)),
                "mse": mse,
                "periods": periods
            }
        except Exception as e:
            return {"error": f"Linear forecast failed: {str(e)}"}
    
    def _create_arima_forecast(self, df: pd.DataFrame, target_col: str, periods: int) -> Dict[str, Any]:
        """Create ARIMA forecast"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            values = df[target_col].values
            
            # Fit ARIMA model (auto-detect parameters)
            model = ARIMA(values, order=(1, 1, 1))
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_result = fitted_model.forecast(steps=periods, alpha=0.05)
            forecasts = forecast_result
            
            return {
                "model": "ARIMA(1,1,1)",
                "forecasts": forecasts.tolist(),
                "historical_values": values.tolist(),
                "periods": periods,
                "aic": fitted_model.aic,
                "bic": fitted_model.bic
            }
        except Exception as e:
            return {"error": f"ARIMA forecast failed: {str(e)}"}
    
    def _create_prophet_forecast(self, df: pd.DataFrame, target_col: str, time_col: str, periods: int) -> Dict[str, Any]:
        """Create Prophet forecast"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            prophet_df = pd.DataFrame()
            if time_col and time_col in df.columns:
                prophet_df['ds'] = pd.to_datetime(df[time_col])
            else:
                # Create synthetic time index
                prophet_df['ds'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
            
            prophet_df['y'] = df[target_col].values
            
            # Fit Prophet model
            model = Prophet()
            model.fit(prophet_df)
            
            # Generate future dates
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            return {
                "model": "Prophet",
                "forecasts": forecast['yhat'].tail(periods).tolist(),
                "confidence_lower": forecast['yhat_lower'].tail(periods).tolist(),
                "confidence_upper": forecast['yhat_upper'].tail(periods).tolist(),
                "historical_values": df[target_col].tolist(),
                "periods": periods,
                "trend": forecast['trend'].tail(periods).tolist()
            }
        except Exception as e:
            return {"error": f"Prophet forecast failed: {str(e)}"}
    
    def _create_exponential_smoothing_forecast(self, df: pd.DataFrame, target_col: str, periods: int) -> Dict[str, Any]:
        """Create Exponential Smoothing forecast"""
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            values = df[target_col].values
            
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(values, trend='add', seasonal=None)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecasts = fitted_model.forecast(periods)
            
            return {
                "model": "Exponential Smoothing",
                "forecasts": forecasts.tolist(),
                "historical_values": values.tolist(),
                "periods": periods
            }
        except Exception as e:
            return {"error": f"Exponential Smoothing forecast failed: {str(e)}"}

# Prescriptive Analysis Agent
class PrescriptiveAnalysisAgent:
    def __init__(self, llm: ChatGroq):
        self.llm = llm
        self.prescriptive_prompt = PromptTemplates.get_prescriptive_analysis_prompt()
    
    def generate_simulation_config(self, user_input: str, query_data: List[Dict], columns: List[str], context: str = "") -> Dict[str, Any]:
        """Generate simulation configuration based on current data"""
        if not query_data or not columns:
            return {"error": "No data available for prescriptive analysis"}
        
        # Fallback configuration
        fallback_config = {
            "simulation_type": "what_if",
            "independent_variables": [columns[0]] if columns else [],
            "dependent_variable": columns[1] if len(columns) > 1 else columns[0] if columns else None,
            "scenarios": [
                {
                    "name": "Baseline",
                    "parameters": {},
                    "description": "Current state without changes"
                }
            ],
            "description": "Basic what-if simulation based on available data"
        }
        
        try:
            prompt = self.prescriptive_prompt.format(
                user_input=user_input,
                query_data=str(query_data[:10]),  # First 10 rows for analysis
                columns=str(columns),
                descriptive_context=context
            )
            
            response = self.llm.invoke(prompt)
            response_content = response.content.strip()
            
            # Parse JSON response with fallback
            if "```json" in response_content:
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    response_content = json_match.group(1).strip()
            
            config = json.loads(response_content)
            
            # Validate and set defaults
            config.setdefault("simulation_type", fallback_config["simulation_type"])
            config.setdefault("independent_variables", fallback_config["independent_variables"])
            config.setdefault("dependent_variable", fallback_config["dependent_variable"])
            config.setdefault("scenarios", fallback_config["scenarios"])
            
            return config
            
        except Exception as e:
            logger.error(f"Error generating simulation config: {e}")
            return fallback_config
    
    def run_simulation(self, query_data: List[Dict], simulation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run simulation based on configuration"""
        try:
            if not query_data:
                return {"error": "No data for simulation"}
            
            df = pd.DataFrame(query_data)
            simulation_type = simulation_config.get("simulation_type", "what_if")
            independent_vars = simulation_config.get("independent_variables", [])
            dependent_var = simulation_config.get("dependent_variable")
            scenarios = simulation_config.get("scenarios", [])
            
            if simulation_type == "what_if":
                return self._run_what_if_simulation(df, independent_vars, dependent_var, scenarios)
            elif simulation_type == "sensitivity":
                return self._run_sensitivity_analysis(df, independent_vars, dependent_var)
            elif simulation_type == "scenario":
                return self._run_scenario_analysis(df, independent_vars, dependent_var, scenarios)
            else:
                return self._run_what_if_simulation(df, independent_vars, dependent_var, scenarios)
                
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            return {"error": f"Simulation failed: {str(e)}"}
    
    def _run_what_if_simulation(self, df: pd.DataFrame, independent_vars: List[str], dependent_var: str, scenarios: List[Dict]) -> Dict[str, Any]:
        """Run what-if simulation"""
        try:
            results = {
                "simulation_type": "What-If Analysis",
                "scenarios": [],
                "baseline": {}
            }
            
            # Calculate baseline
            if dependent_var and dependent_var in df.columns:
                baseline_value = df[dependent_var].mean()
                results["baseline"] = {
                    "scenario": "Current State",
                    "value": baseline_value,
                    "description": f"Current average {dependent_var}"
                }
            
            # Run scenarios
            for scenario in scenarios:
                scenario_name = scenario.get("name", "Scenario")
                parameters = scenario.get("parameters", {})
                
                # Create modified dataset
                modified_df = df.copy()
                for param, value in parameters.items():
                    if param in modified_df.columns:
                        # Apply percentage change if value is string with %
                        if isinstance(value, str) and '%' in value:
                            percent_change = float(value.replace('%', '')) / 100
                            modified_df[param] = modified_df[param] * (1 + percent_change)
                        else:
                            try:
                                modified_df[param] = float(value)
                            except:
                                pass  # Skip if cannot convert
                
                # Calculate impact on dependent variable
                if dependent_var and dependent_var in modified_df.columns:
                    new_value = modified_df[dependent_var].mean()
                    change = new_value - baseline_value
                    change_percent = (change / baseline_value * 100) if baseline_value != 0 else 0
                    
                    results["scenarios"].append({
                        "name": scenario_name,
                        "parameters": parameters,
                        "result": new_value,
                        "change": change,
                        "change_percent": change_percent,
                        "description": scenario.get("description", "")
                    })
            
            return results
            
        except Exception as e:
            return {"error": f"What-if simulation failed: {str(e)}"}
    
    def _run_sensitivity_analysis(self, df: pd.DataFrame, independent_vars: List[str], dependent_var: str) -> Dict[str, Any]:
        """Run sensitivity analysis"""
        try:
            results = {
                "simulation_type": "Sensitivity Analysis",
                "variables": [],
                "correlations": {}
            }
            
            if dependent_var and dependent_var in df.columns:
                for var in independent_vars:
                    if var in df.columns:
                        # Calculate correlation
                        correlation = df[var].corr(df[dependent_var])
                        
                        # Test different values (+-10%, +-20%)
                        test_values = [-20, -10, 10, 20]  # percentage changes
                        sensitivity_results = []
                        
                        baseline = df[dependent_var].mean()
                        
                        for change_percent in test_values:
                            modified_df = df.copy()
                            modified_df[var] = modified_df[var] * (1 + change_percent/100)
                            new_dependent = modified_df[dependent_var].mean()
                            impact = new_dependent - baseline
                            
                            sensitivity_results.append({
                                "change_percent": change_percent,
                                "impact": impact,
                                "new_value": new_dependent
                            })
                        
                        results["variables"].append({
                            "variable": var,
                            "correlation": correlation,
                            "sensitivity": sensitivity_results
                        })
                        
                        results["correlations"][var] = correlation
            
            return results
            
        except Exception as e:
            return {"error": f"Sensitivity analysis failed: {str(e)}"}
    
    def _run_scenario_analysis(self, df: pd.DataFrame, independent_vars: List[str], dependent_var: str, scenarios: List[Dict]) -> Dict[str, Any]:
        """Run comprehensive scenario analysis"""
        # For now, use what-if simulation logic
        return self._run_what_if_simulation(df, independent_vars, dependent_var, scenarios)

# Main Modular SQL Agent with Enhanced Features
class SQLAgent:
    def __init__(self):
        # Check if required configuration is available
        logger.info(Config.GROQ_API_KEY)
        if not Config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured. Please set it in your environment variables.")
        
        self.db_manager = DatabaseManager()
        self.schema_inspector = SchemaInspector(self.db_manager)
        self.llm = ChatGroq(
            groq_api_key=Config.GROQ_API_KEY,
            model_name=Config.GROQ_MODEL,
            temperature=0.1
        )
        
        # Initialize all sub-agents
        self.query_generator = SQLQueryGenerator(
            self.llm, 
            self.schema_inspector.get_schema_info()
        )
        self.query_executor = QueryExecutor(self.db_manager)
        self.visualization_agent = VisualizationAgent(self.llm)
        self.reflection_agent = ReflectionAgent(self.llm)
        self.predictive_agent = PredictiveAnalysisAgent(self.llm)
        self.prescriptive_agent = PrescriptiveAnalysisAgent(self.llm)
        
        # Router prompt
        self.router_prompt = PromptTemplates.get_router_prompt()
        self.general_prompt = PromptTemplates.get_general_response_prompt()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("route_query", self.route_query)
        workflow.add_node("generate_sql", self.generate_sql)
        workflow.add_node("execute_sql", self.execute_sql)
        workflow.add_node("reflect_response", self.reflect_response)
        workflow.add_node("refine_response", self.refine_response)
        workflow.add_node("generate_visualization", self.generate_visualization)
        workflow.add_node("generate_predictive", self.generate_predictive)
        workflow.add_node("generate_prescriptive", self.generate_prescriptive)
        workflow.add_node("format_sql_response", self.format_sql_response)
        workflow.add_node("handle_general", self.handle_general)
        workflow.add_node("finalize_response", self.finalize_response)
        
        # Add edges
        workflow.set_entry_point("route_query")
        workflow.add_conditional_edges(
            "route_query",
            self.route_decision,
            {
                "sql_query": "generate_sql",
                "visualization": "generate_sql", 
                "predictive": "generate_sql",
                "prescriptive": "generate_sql",
                "general": "handle_general"
            }
        )
        
        workflow.add_edge("generate_sql", "execute_sql")
        workflow.add_edge("execute_sql", "format_sql_response")
        workflow.add_edge("format_sql_response", "reflect_response")
        workflow.add_edge("reflect_response", "refine_response")
        
        workflow.add_conditional_edges(
            "refine_response",
            self.check_analysis_needed,
            {
                "visualization": "generate_visualization",
                "predictive": "generate_predictive",
                "prescriptive": "generate_prescriptive",
                "complete": "finalize_response"
            }
        )
        
        workflow.add_edge("generate_visualization", "finalize_response")
        workflow.add_edge("generate_predictive", "finalize_response")
        workflow.add_edge("generate_prescriptive", "finalize_response")
        workflow.add_edge("handle_general", "finalize_response")
        workflow.add_edge("finalize_response", END)
        
        return workflow.compile()
    
    def route_query(self, state: AgentState) -> AgentState:
        """Route the query to appropriate handler"""
        user_input = state["user_input"]
        
        # Get session context
        context = ""
        if state["session_id"]:
            session_data = self.db_manager.get_session_context(state["session_id"])
            context = "\n".join([f"User: {s['user_message']}\nBot: {s['bot_response']}" for s in session_data])
        
        # Enhanced routing logic to detect multiple query types
        query_types = []
        user_input_lower = user_input.lower()
        
        # Check for visualization keywords
        viz_keywords = ["chart", "graph", "plot", "visualize", "visualization", "show chart", "bar chart", "pie chart"]
        has_viz_keyword = any(keyword in user_input_lower for keyword in viz_keywords)
        
        # Check for SQL/data keywords
        sql_keywords = ["count", "how many", "show", "list", "average", "sum", "total", "find", "data", "select", "table"]
        has_sql_keyword = any(keyword in user_input_lower for keyword in sql_keywords)
        
        # Check for predictive analysis keywords
        predictive_keywords = ["forecast", "predict", "prediction", "future", "trend", "project", "estimate", "anticipate", "expect", "model", "arima", "prophet"]
        has_predictive_keyword = any(keyword in user_input_lower for keyword in predictive_keywords)
        
        # Check for prescriptive analysis keywords
        prescriptive_keywords = ["simulate", "simulation", "what if", "scenario", "optimize", "recommend", "suggest", "best", "improve", "change", "modify", "test"]
        has_prescriptive_keyword = any(keyword in user_input_lower for keyword in prescriptive_keywords)
        
        # Check for general keywords
        general_keywords = ["hello", "hi", "help", "what can you", "greeting"]
        has_general_keyword = any(keyword in user_input_lower for keyword in general_keywords)
        
        # Determine query types based on keywords
        if has_general_keyword and not (has_viz_keyword or has_sql_keyword or has_predictive_keyword or has_prescriptive_keyword):
            query_types = ["general"]
        elif has_predictive_keyword:
            query_types = ["sql_query", "predictive"]
            if has_viz_keyword:
                query_types.append("visualization")
        elif has_prescriptive_keyword:
            query_types = ["sql_query", "prescriptive"]
            if has_viz_keyword:
                query_types.append("visualization")
        elif has_viz_keyword and (has_sql_keyword or any(word in user_input_lower for word in ["data", "show", "display"])):
            query_types = ["sql_query", "visualization"]
        elif has_viz_keyword:
            query_types = ["sql_query", "visualization"]  # Visualization typically needs data first
        elif has_sql_keyword:
            query_types = ["sql_query"]
        else:
            query_types = ["general"]
        
        # Fallback to LLM if uncertain
        if len(query_types) == 0 or (len(query_types) == 1 and query_types[0] == "general" and (has_sql_keyword or has_viz_keyword)):
            try:
                # Use LLM to route the query as fallback
                prompt = self.router_prompt.format(
                    user_input=user_input,
                    context=context
                )
                
                response = self.llm.invoke(prompt)
                response_content = response.content.strip()
                
                # Try to parse JSON response
                import json
                try:
                    parsed_types = json.loads(response_content)
                    if isinstance(parsed_types, list):
                        query_types = parsed_types
                    else:
                        query_types = [parsed_types] if parsed_types in ["sql_query", "visualization", "general"] else ["general"]
                except:
                    # Fallback to single type parsing
                    query_type = response_content.lower()
                    if query_type in ["sql_query", "visualization", "general"]:
                        query_types = [query_type]
                    else:
                        query_types = ["general"]
            except:
                query_types = ["general"]
        
        state["query_types"] = query_types
        state["context"] = context
        return state
    
    def route_decision(self, state: AgentState) -> str:
        """Decide the route based on query types"""
        query_types = state["query_types"]
        
        # Priority routing logic
        if "general" in query_types and len(query_types) == 1:
            return "general"
        elif "predictive" in query_types:
            return "predictive"  # Predictive analysis needs data first
        elif "prescriptive" in query_types:
            return "prescriptive"  # Prescriptive analysis needs data first
        elif "sql_query" in query_types:
            return "sql_query"  # Start with SQL generation if data is needed
        else:
            return "general"
    
    def generate_sql(self, state: AgentState) -> AgentState:
        """Generate SQL query"""
        sql_query = self.query_generator.generate_query(
            state["user_input"], 
            state["context"]
        )
        state["sql_query"] = sql_query
        state["generated_sql"] = sql_query
        return state
    
    def execute_sql(self, state: AgentState) -> AgentState:
        """Execute SQL query"""
        if state["sql_query"] == "UNSUPPORTED_QUERY":
            state["query_results"] = {
                "success": False,
                "error": "Query not supported. Only simple SELECT queries are allowed.",
                "data": [],
                "columns": [],
                "row_count": 0
            }
            state["query_data"] = []
        else:
            query_results = self.query_executor.execute_query(state["sql_query"])
            state["query_results"] = query_results
            # Extract only the data for the new query_data variable
            state["query_data"] = query_results.get("data", []) if query_results.get("success") else []
        
        state["data_query_result"] = state["query_results"]
        return state
    
    def format_sql_response(self, state: AgentState) -> AgentState:
        """Format the SQL response"""
        results = state["query_results"]
        
        if not results["success"]:
            response = f"Sorry, I couldn't process your query: {results['error']}"
        else:
            if results["row_count"] == 0:
                response = "No data found matching your query."
            else:
                response = self._format_data_response(results)
        
        state["final_response"] = response
        return state
    
    def reflect_response(self, state: AgentState) -> AgentState:
        """Reflect on the response quality"""
        if state["query_results"]["success"]:
            feedback = self.reflection_agent.reflect_on_response(
                state["user_input"],
                state["sql_query"],
                state["query_results"],
                state["final_response"]
            )
            state["reflection_feedback"] = feedback
        else:
            state["reflection_feedback"] = "APPROVED"
        return state
    
    def refine_response(self, state: AgentState) -> AgentState:
        """Refine the response based on reflection"""
        if state["reflection_feedback"] != "APPROVED":
            refined_response = self.reflection_agent.refine_response(
                state["user_input"],
                state["sql_query"],
                state["query_results"],
                state["final_response"],
                state["reflection_feedback"]
            )
            state["refined_response"] = refined_response
            state["final_response"] = refined_response
        else:
            state["refined_response"] = state["final_response"]
        return state
    
    def check_analysis_needed(self, state: AgentState) -> str:
        """Check what type of analysis is needed"""
        query_types = state["query_types"]
        
        # Priority order: predictive > prescriptive > visualization > complete
        if "predictive" in query_types:
            return "predictive"
        elif "prescriptive" in query_types:
            return "prescriptive"
        elif "visualization" in query_types:
            return "visualization"
        else:
            return "complete"
    
    def generate_visualization(self, state: AgentState) -> AgentState:
        """Generate visualization"""
        if state["query_results"]["success"] and state["query_results"]["data"]:
            # Generate chart configuration
            chart_config = self.visualization_agent.generate_chart_config(
                state["user_input"],
                state["query_results"]
            )
            
            if "error" not in chart_config:
                # Create visualization
                viz_base64 = self.visualization_agent.create_visualization(
                    state["query_results"],
                    chart_config
                )
                state["visualization_result"] = viz_base64
                state["chart_type"] = chart_config.get("chart_type", "bar")
                state["visualization_data"] = chart_config
            else:
                state["visualization_result"] = None
                state["final_response"] += f"\n\nVisualization Error: {chart_config['error']}"
        else:
            state["visualization_result"] = None
        
        return state
    
    def generate_predictive(self, state: AgentState) -> AgentState:
        """Generate predictive analysis"""
        if state["query_results"]["success"] and state["query_results"]["data"]:
            # Determine analysis type
            state["analysis_type"] = "predictive"
            
            # Generate forecast configuration
            forecast_config = self.predictive_agent.generate_forecast_config(
                state["user_input"],
                state["query_data"],
                state["query_results"]["columns"],
                state["context"]
            )
            
            if "error" not in forecast_config:
                # Create forecast
                forecast_results = self.predictive_agent.create_forecast(
                    state["query_data"],
                    forecast_config
                )
                state["forecast_results"] = forecast_results
                state["predictive_model"] = forecast_config.get("model_type", "linear")
                
                # Update response with predictive insights
                if "error" not in forecast_results:
                    forecast_summary = f"\n\n## Predictive Analysis\n"
                    forecast_summary += f"**Model Used:** {forecast_results.get('model', 'Unknown')}\n"
                    forecast_summary += f"**Forecast Periods:** {forecast_results.get('periods', 'N/A')}\n"
                    
                    if 'forecasts' in forecast_results:
                        forecasts = forecast_results['forecasts']
                        if len(forecasts) > 0:
                            forecast_summary += f"**Next Period Forecast:** {forecasts[0]:.2f}\n"
                            if len(forecasts) > 1:
                                forecast_summary += f"**Future Values:** {', '.join([f'{f:.2f}' for f in forecasts[:5]])}\n"
                    
                    if 'mae' in forecast_results:
                        forecast_summary += f"**Model Accuracy (MAE):** {forecast_results['mae']:.2f}\n"
                    
                    state["final_response"] += forecast_summary
                else:
                    state["final_response"] += f"\n\nPredictive Analysis Error: {forecast_results['error']}"
            else:
                state["final_response"] += f"\n\nPredictive Analysis Error: {forecast_config['error']}"
        else:
            state["forecast_results"] = None
        
        return state
    
    def generate_prescriptive(self, state: AgentState) -> AgentState:
        """Generate prescriptive analysis"""
        if state["query_results"]["success"] and state["query_results"]["data"]:
            # Determine analysis type
            state["analysis_type"] = "prescriptive"
            
            # Generate simulation configuration
            simulation_config = self.prescriptive_agent.generate_simulation_config(
                state["user_input"],
                state["query_data"],
                state["query_results"]["columns"],
                state["context"]
            )
            
            if "error" not in simulation_config:
                # Run simulation
                simulation_results = self.prescriptive_agent.run_simulation(
                    state["query_data"],
                    simulation_config
                )
                state["simulation_results"] = simulation_results
                state["simulation_parameters"] = simulation_config
                
                # Update response with prescriptive insights
                if "error" not in simulation_results:
                    simulation_summary = f"\n\n## Prescriptive Analysis\n"
                    simulation_summary += f"**Analysis Type:** {simulation_results.get('simulation_type', 'Unknown')}\n"
                    
                    if 'baseline' in simulation_results:
                        baseline = simulation_results['baseline']
                        simulation_summary += f"**Baseline Value:** {baseline.get('value', 'N/A')}\n"
                    
                    if 'scenarios' in simulation_results:
                        scenarios = simulation_results['scenarios']
                        simulation_summary += f"**Scenarios Analyzed:** {len(scenarios)}\n"
                        
                        for i, scenario in enumerate(scenarios[:3]):  # Show top 3 scenarios
                            name = scenario.get('name', f'Scenario {i+1}')
                            result = scenario.get('result', 'N/A')
                            change_percent = scenario.get('change_percent', 0)
                            simulation_summary += f"- **{name}:** {result:.2f} ({change_percent:+.1f}%)\n"
                    
                    if 'variables' in simulation_results:
                        variables = simulation_results['variables']
                        simulation_summary += f"**Key Variables:** {len(variables)}\n"
                        for var in variables[:3]:  # Show top 3 variables
                            var_name = var.get('variable', 'Unknown')
                            correlation = var.get('correlation', 0)
                            simulation_summary += f"- **{var_name}:** Correlation = {correlation:.3f}\n"
                    
                    state["final_response"] += simulation_summary
                else:
                    state["final_response"] += f"\n\nPrescriptive Analysis Error: {simulation_results['error']}"
            else:
                state["final_response"] += f"\n\nPrescriptive Analysis Error: {simulation_config['error']}"
        else:
            state["simulation_results"] = None
        
        return state

    def handle_general(self, state: AgentState) -> AgentState:
        """Handle general queries"""
        available_tables = list(self.schema_inspector.get_schema_info().keys())
        
        prompt = self.general_prompt.format(
            user_input=state["user_input"],
            available_tables=", ".join(available_tables),
            context=state["context"]
        )
        
        response = self.llm.invoke(prompt)
        state["final_response"] = response.content.strip()
        return state
    
    def finalize_response(self, state: AgentState) -> AgentState:
        """Finalize the response with markdown formatting"""
        state["markdown_result"] = self._format_markdown_response(state)
        return state
    
    def _format_data_response(self, results: Dict[str, Any]) -> str:
        """Format data results into a readable response"""
        data = results["data"]
        columns = results["columns"]
        
        if len(data) == 1 and len(columns) == 1:
            # Single value result
            return f"Result: {data[0][columns[0]]}"
        else:
            # Multiple rows/columns - create table
            response_parts = [f"Found {results['row_count']} result(s):"]
            
            if data and columns:
                # Create markdown table
                header = f"| {' | '.join(columns)} |"
                separator = f"| {' | '.join(['---'] * len(columns))} |"
                response_parts.append(header)
                response_parts.append(separator)

                for row in data[:10]:  # Limit to 10 rows
                    row_str = [str(row[col]) for col in columns]
                    response_parts.append(f"| {' | '.join(row_str)} |")
                
                if results['row_count'] > 10:
                    response_parts.append(f"\n... and {results['row_count'] - 10} more rows.")

                return "\n".join(response_parts)
            else:
                return "No data found or columns are missing."
    
    def _format_markdown_response(self, state: AgentState) -> str:
        """Format the complete response in markdown"""
        markdown_parts = []
        
        # Add main response
        markdown_parts.append(f"## Response\n{state['final_response']}")
        
        # Add SQL query if available
        if state.get("generated_sql") and state["generated_sql"] != "UNSUPPORTED_QUERY":
            markdown_parts.append(f"\n## Generated SQL\n```sql\n{state['generated_sql']}\n```")
        
        # Add reflection feedback if available
        if state.get("reflection_feedback") and state["reflection_feedback"] != "APPROVED":
            markdown_parts.append(f"\n## Analysis Notes\n{state['reflection_feedback']}")
        
        # Add visualization info if available
        if state.get("visualization_result"):
            markdown_parts.append(f"\n## Visualization\nChart Type: {state.get('chart_type', 'Unknown')}")
        
        return "\n".join(markdown_parts)
    
    def process_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a message and return structured response"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        initial_state = AgentState(
            messages=[HumanMessage(content=message)],
            user_input=message,
            session_id=session_id,
            query_types=[],  # Changed from query_type to query_types
            context="",
            sql_query="",
            query_results={},
            query_data=[],  # New field for only query result data
            final_response="",
            generated_sql=None,
            data_query_result=None,
            markdown_result=None,
            visualization_result=None,
            reflection_feedback=None,
            refined_response=None,
            chart_type=None,
            visualization_data=None,
            # New fields for predictive and prescriptive analysis
            analysis_type=None,
            forecast_results=None,
            simulation_results=None,
            predictive_model=None,
            simulation_parameters=None
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Save to session
        query_type_str = ", ".join(final_state["query_types"]) if final_state["query_types"] else "general"
        self.db_manager.save_session(
            session_id,
            message,
            final_state["final_response"],
            query_type_str
        )
        
        # Return structured response
        return {
            "response": final_state["final_response"],
            "session_id": session_id,
            "query_types": final_state["query_types"],  # Return as array
            "query_type": query_type_str,  # Keep for backward compatibility
            "generated_sql": final_state.get("generated_sql"),
            "data_query_result": final_state.get("data_query_result"),
            "markdown_result": final_state.get("markdown_result"),
            "visualization_result": final_state.get("visualization_result"),
            # New analysis results
            "analysis_type": final_state.get("analysis_type"),
            "forecast_results": final_state.get("forecast_results"),
            "simulation_results": final_state.get("simulation_results"),
            "predictive_model": final_state.get("predictive_model"),
            "simulation_parameters": final_state.get("simulation_parameters")
        }

def show_db_config_form():
    st.header("Configure Database Connection")
    with st.form("db_config_form"):
        host = st.text_input("Host", value="localhost")
        port = st.number_input("Port", value=5432)
        username = st.text_input("Username", value="user")
        password = st.text_input("Password", type="password", value="password")
        database = st.text_input("Database", value="dbname")
        
        submitted = st.form_submit_button("Connect")
        if submitted:
            postgres_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
            try:
                with st.spinner("Connecting to database..."):
                    conn = psycopg2.connect(postgres_url)
                    conn.close()
                
                st.success("Database connection successful!")
                Config.POSTGRES_URL = postgres_url
                st.session_state.db_connected = True
                st.session_state.agent = SQLAgent()
                st.rerun()

            except Exception as e:
                st.error(f"Database connection failed: {e}")

def show_chat_interface():
    st.title("Advanced Analytics SQL Agent - Descriptive, Predictive & Prescriptive Analysis")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display structured response
                if "response" in message:
                    st.markdown(message["response"])
                
                # Display SQL query if available
                if message.get("generated_sql") and message["generated_sql"] != "UNSUPPORTED_QUERY":
                    with st.expander("Generated SQL Query"):
                        st.code(message["generated_sql"], language="sql")
                
                # Display data results if available
                if message.get("data_query_result") and message["data_query_result"].get("success"):
                    data_result = message["data_query_result"]
                    if data_result.get("data"):
                        with st.expander("Query Results"):
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(data_result["data"])
                            st.dataframe(df)
                            st.caption(f"Total rows: {data_result['row_count']}")
                
                # Display visualization if available
                if message.get("visualization_result"):
                    with st.expander("Visualization"):
                        # Decode base64 image
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        image_data = base64.b64decode(message["visualization_result"])
                        image = Image.open(BytesIO(image_data))
                        st.image(image, caption="Data Visualization", use_column_width=True)
                
                # Display predictive analysis if available
                if message.get("forecast_results") and "error" not in message["forecast_results"]:
                    with st.expander("Predictive Analysis Results"):
                        forecast_results = message["forecast_results"]
                        st.subheader(f"Model: {forecast_results.get('model', 'Unknown')}")
                        
                        if 'forecasts' in forecast_results:
                            st.write("**Forecasted Values:**")
                            forecasts = forecast_results['forecasts'][:10]  # Show first 10
                            forecast_df = pd.DataFrame({
                                'Period': range(1, len(forecasts) + 1),
                                'Forecast': forecasts
                            })
                            st.dataframe(forecast_df)
                        
                        if 'mae' in forecast_results:
                            st.metric("Model Accuracy (MAE)", f"{forecast_results['mae']:.2f}")
                        
                        if 'historical_values' in forecast_results:
                            st.write("**Historical vs Forecasted Trend:**")
                            historical = forecast_results['historical_values']
                            forecasts = forecast_results.get('forecasts', [])
                            
                            # Create trend visualization
                            fig, ax = plt.subplots(figsize=(10, 4))
                            hist_x = range(len(historical))
                            forecast_x = range(len(historical), len(historical) + len(forecasts))
                            
                            ax.plot(hist_x, historical, label='Historical', marker='o')
                            ax.plot(forecast_x, forecasts, label='Forecast', marker='s', linestyle='--')
                            ax.legend()
                            ax.set_title('Historical Data vs Forecast')
                            st.pyplot(fig)
                            plt.close()
                
                # Display prescriptive analysis if available
                if message.get("simulation_results") and "error" not in message["simulation_results"]:
                    with st.expander("Prescriptive Analysis Results"):
                        simulation_results = message["simulation_results"]
                        st.subheader(f"Analysis: {simulation_results.get('simulation_type', 'Unknown')}")
                        
                        if 'baseline' in simulation_results:
                            baseline = simulation_results['baseline']
                            st.metric("Baseline Value", f"{baseline.get('value', 'N/A')}")
                        
                        if 'scenarios' in simulation_results:
                            st.write("**Scenario Analysis:**")
                            scenarios_data = []
                            for scenario in simulation_results['scenarios']:
                                scenarios_data.append({
                                    'Scenario': scenario.get('name', 'Unknown'),
                                    'Result': scenario.get('result', 0),
                                    'Change (%)': f"{scenario.get('change_percent', 0):+.1f}%",
                                    'Description': scenario.get('description', '')
                                })
                            
                            if scenarios_data:
                                scenarios_df = pd.DataFrame(scenarios_data)
                                st.dataframe(scenarios_df)
                        
                        if 'variables' in simulation_results:
                            st.write("**Variable Sensitivity:**")
                            variables_data = []
                            for var in simulation_results['variables']:
                                variables_data.append({
                                    'Variable': var.get('variable', 'Unknown'),
                                    'Correlation': f"{var.get('correlation', 0):.3f}"
                                })
                            
                            if variables_data:
                                variables_df = pd.DataFrame(variables_data)
                                st.dataframe(variables_df)
                
                # Display markdown formatted response
                if message.get("markdown_result"):
                    with st.expander("Detailed Analysis"):
                        st.markdown(message["markdown_result"])
            else:
                st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask questions, request visualizations, get forecasts, or run simulations on your data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing your request..."):
                agent = st.session_state.agent
                result = agent.process_message(prompt, st.session_state.session_id)
                
                # Display main response
                st.markdown(result["response"])
                
                # Display SQL query if available
                if result.get("generated_sql") and result["generated_sql"] != "UNSUPPORTED_QUERY":
                    with st.expander("Generated SQL Query"):
                        st.code(result["generated_sql"], language="sql")
                
                # Display data results if available
                if result.get("data_query_result") and result["data_query_result"].get("success"):
                    data_result = result["data_query_result"]
                    if data_result.get("data"):
                        with st.expander("Query Results"):
                            # Convert to DataFrame for better display
                            df = pd.DataFrame(data_result["data"])
                            st.dataframe(df)
                            st.caption(f"Total rows: {data_result['row_count']}")
                
                # Display visualization if available
                if result.get("visualization_result"):
                    with st.expander("Visualization"):
                        # Decode base64 image
                        import base64
                        from io import BytesIO
                        from PIL import Image
                        
                        image_data = base64.b64decode(result["visualization_result"])
                        image = Image.open(BytesIO(image_data))
                        st.image(image, caption="Data Visualization", use_column_width=True)
                
                # Display markdown formatted response
                if result.get("markdown_result"):
                    with st.expander("Detailed Analysis"):
                        st.markdown(result["markdown_result"])
                
                # Display predictive analysis if available
                if result.get("forecast_results") and "error" not in result["forecast_results"]:
                    with st.expander("Predictive Analysis Results"):
                        forecast_results = result["forecast_results"]
                        st.subheader(f"Model: {forecast_results.get('model', 'Unknown')}")
                        
                        if 'forecasts' in forecast_results:
                            st.write("**Forecasted Values:**")
                            forecasts = forecast_results['forecasts'][:10]  # Show first 10
                            forecast_df = pd.DataFrame({
                                'Period': range(1, len(forecasts) + 1),
                                'Forecast': forecasts
                            })
                            st.dataframe(forecast_df)
                        
                        if 'mae' in forecast_results:
                            st.metric("Model Accuracy (MAE)", f"{forecast_results['mae']:.2f}")
                
                # Display prescriptive analysis if available
                if result.get("simulation_results") and "error" not in result["simulation_results"]:
                    with st.expander("Prescriptive Analysis Results"):
                        simulation_results = result["simulation_results"]
                        st.subheader(f"Analysis: {simulation_results.get('simulation_type', 'Unknown')}")
                        
                        if 'baseline' in simulation_results:
                            baseline = simulation_results['baseline']
                            st.metric("Baseline Value", f"{baseline.get('value', 'N/A')}")
                        
                        if 'scenarios' in simulation_results:
                            st.write("**Scenario Analysis:**")
                            scenarios_data = []
                            for scenario in simulation_results['scenarios']:
                                scenarios_data.append({
                                    'Scenario': scenario.get('name', 'Unknown'),
                                    'Result': scenario.get('result', 0),
                                    'Change (%)': f"{scenario.get('change_percent', 0):+.1f}%"
                                })
                            
                            if scenarios_data:
                                scenarios_df = pd.DataFrame(scenarios_data)
                                st.dataframe(scenarios_df)
        
        # Save the complete result to session
        st.session_state.messages.append({
            "role": "assistant",
            "response": result["response"],
            "generated_sql": result.get("generated_sql"),
            "data_query_result": result.get("data_query_result"),
            "markdown_result": result.get("markdown_result"),
            "visualization_result": result.get("visualization_result"),
            "forecast_results": result.get("forecast_results"),
            "simulation_results": result.get("simulation_results"),
            "analysis_type": result.get("analysis_type")
        })

def main():
    st.set_page_config(page_title="Bisee Chatbot", layout="wide")

    if "db_connected" not in st.session_state:
        st.session_state.db_connected = False

    if not st.session_state.db_connected:
        show_db_config_form()
    else:
        show_chat_interface()

if __name__ == "__main__":
    main()