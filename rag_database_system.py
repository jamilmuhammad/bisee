import os
import json
import uuid
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

# Database and Vector Store
import psycopg2
from psycopg2.extras import RealDictCursor
import pandas as pd
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# LLM Integration
import groq
from groq import Groq

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configuration
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(Enum):
    DATABASE_INSIGHT = "database_insight"
    SQL_GENERATOR = "sql_generator"
    VISUALIZATION = "visualization"
    RECOMMENDATION = "recommendation"
    ORCHESTRATOR = "orchestrator"

@dataclass
class SessionContext:
    session_id: str
    user_id: str
    database_config: Dict
    conversation_history: List[Dict]
    insights_generated: List[Dict]
    current_context: Dict
    timestamp: datetime

@dataclass
class InsightResult:
    insight_id: str
    insight_type: str
    content: Dict
    metadata: Dict
    confidence_score: float
    timestamp: datetime

class DatabaseSchemaExtractor:
    """Enhanced database schema extraction with deep learning capabilities"""
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
        self.schema_cache = {}
        
    async def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.db_config['host'],
                port=self.db_config['port'],
                database=self.db_config['database'],
                user=self.db_config['user'],
                password=self.db_config['password'],
                cursor_factory=RealDictCursor
            )
            logger.info(f"Connected to database: {self.db_config['database']}")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    async def extract_complete_schema(self) -> Dict:
        """Extract complete database schema with relationships"""
        if not self.connection:
            await self.connect()
        
        cursor = self.connection.cursor()
        
        schema_info = {
            'tables': {},
            'relationships': [],
            'indexes': {},
            'constraints': {},
            'functions': [],
            'views': [],
            'materialized_views': [],
            'metadata': {}
        }
        
        # Extract all tables
        cursor.execute("""
            SELECT table_name, table_type, table_schema
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name
        """)
        
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table['table_name']
            table_info = await self._extract_table_details(table_name)
            schema_info['tables'][table_name] = table_info
        
        # Extract relationships
        schema_info['relationships'] = await self._extract_relationships()
        
        # Extract indexes
        schema_info['indexes'] = await self._extract_indexes()
        
        # Extract constraints
        schema_info['constraints'] = await self._extract_constraints()
        
        # Extract functions and procedures
        schema_info['functions'] = await self._extract_functions()
        
        # Extract views
        schema_info['views'] = await self._extract_views()
        
        # Generate semantic understanding
        schema_info['semantic_analysis'] = await self._generate_semantic_analysis(schema_info)
        
        cursor.close()
        return schema_info
    
    async def _extract_table_details(self, table_name: str) -> Dict:
        """Extract detailed information about a table"""
        cursor = self.connection.cursor()
        
        # Column information
        cursor.execute("""
            SELECT 
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale,
                c.ordinal_position,
                CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key,
                CASE WHEN uk.column_name IS NOT NULL THEN true ELSE false END as is_unique
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'PRIMARY KEY'
                    AND tc.table_name = %s
            ) pk ON c.column_name = pk.column_name
            LEFT JOIN (
                SELECT kcu.column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.constraint_type = 'UNIQUE'
                    AND tc.table_name = %s
            ) uk ON c.column_name = uk.column_name
            WHERE c.table_name = %s
            ORDER BY c.ordinal_position
        """, (table_name, table_name, table_name))
        
        columns = cursor.fetchall()
        
        # Get row count and sample data
        cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        row_count = cursor.fetchone()['count']
        
        # Get sample data for analysis
        cursor.execute(f'SELECT * FROM "{table_name}" LIMIT 10')
        sample_data = cursor.fetchall()
        
        # Analyze data patterns
        data_patterns = await self._analyze_data_patterns(table_name, columns)
        
        table_info = {
            'name': table_name,
            'columns': [dict(col) for col in columns],
            'row_count': row_count,
            'sample_data': [dict(row) for row in sample_data],
            'data_patterns': data_patterns,
            'business_context': await self._infer_business_context(table_name, columns)
        }
        
        cursor.close()
        return table_info
    
    async def _analyze_data_patterns(self, table_name: str, columns: List[Dict]) -> Dict:
        """Analyze data patterns in the table"""
        cursor = self.connection.cursor()
        patterns = {}
        
        for col in columns:
            col_name = col['column_name']
            data_type = col['data_type']
            
            # Basic statistics
            if data_type in ['integer', 'bigint', 'numeric', 'real', 'double precision']:
                cursor.execute(f"""
                    SELECT 
                        MIN("{col_name}") as min_val,
                        MAX("{col_name}") as max_val,
                        AVG("{col_name}") as avg_val,
                        STDDEV("{col_name}") as std_val,
                        COUNT(DISTINCT "{col_name}") as unique_count
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'numeric',
                        'statistics': dict(result),
                        'distribution': 'normal'  # Could be enhanced with actual distribution analysis
                    }
            
            elif data_type in ['character varying', 'text', 'character']:
                cursor.execute(f"""
                    SELECT 
                        COUNT(DISTINCT "{col_name}") as unique_count,
                        AVG(LENGTH("{col_name}")) as avg_length,
                        MAX(LENGTH("{col_name}")) as max_length
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'categorical',
                        'statistics': dict(result),
                        'cardinality': 'high' if result['unique_count'] > 100 else 'low'
                    }
            
            elif data_type in ['timestamp', 'date', 'time']:
                cursor.execute(f"""
                    SELECT 
                        MIN("{col_name}") as min_date,
                        MAX("{col_name}") as max_date,
                        COUNT(DISTINCT DATE("{col_name}")) as unique_dates
                    FROM "{table_name}"
                    WHERE "{col_name}" IS NOT NULL
                """)
                result = cursor.fetchone()
                if result:
                    patterns[col_name] = {
                        'type': 'temporal',
                        'statistics': dict(result),
                        'time_series': True
                    }
        
        cursor.close()
        return patterns
    
    async def _infer_business_context(self, table_name: str, columns: List[Dict]) -> Dict:
        """Infer business context from table and column names"""
        # Business domain inference based on naming patterns
        business_domains = {
            'financial': ['transaction', 'payment', 'account', 'balance', 'amount', 'price', 'cost'],
            'customer': ['customer', 'client', 'user', 'person', 'contact', 'profile'],
            'product': ['product', 'item', 'inventory', 'stock', 'catalog'],
            'sales': ['sale', 'order', 'purchase', 'revenue', 'profit'],
            'marketing': ['campaign', 'advertisement', 'promotion', 'lead'],
            'operations': ['operation', 'process', 'workflow', 'status'],
            'analytics': ['metric', 'measure', 'kpi', 'score', 'rating']
        }
        
        table_lower = table_name.lower()
        column_names = [col['column_name'].lower() for col in columns]
        
        domain_scores = {}
        for domain, keywords in business_domains.items():
            score = 0
            for keyword in keywords:
                if keyword in table_lower:
                    score += 2
                for col_name in column_names:
                    if keyword in col_name:
                        score += 1
            domain_scores[domain] = score
        
        primary_domain = max(domain_scores, key=domain_scores.get) if domain_scores else 'general'
        
        return {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores,
            'table_purpose': await self._infer_table_purpose(table_name, columns),
            'key_metrics': await self._identify_key_metrics(columns)
        }
    
    async def _infer_table_purpose(self, table_name: str, columns: List[Dict]) -> str:
        """Infer the purpose of the table"""
        column_names = [col['column_name'].lower() for col in columns]
        
        if any('transaction' in name for name in column_names + [table_name.lower()]):
            return 'transaction_log'
        elif any('customer' in name for name in column_names + [table_name.lower()]):
            return 'customer_data'
        elif any('product' in name for name in column_names + [table_name.lower()]):
            return 'product_catalog'
        elif any('order' in name for name in column_names + [table_name.lower()]):
            return 'order_management'
        else:
            return 'data_storage'
    
    async def _identify_key_metrics(self, columns: List[Dict]) -> List[str]:
        """Identify key metrics columns"""
        metric_indicators = ['amount', 'total', 'count', 'sum', 'average', 'rate', 'score', 'value']
        key_metrics = []
        
        for col in columns:
            col_name = col['column_name'].lower()
            if any(indicator in col_name for indicator in metric_indicators):
                key_metrics.append(col['column_name'])
        
        return key_metrics
    
    async def _extract_relationships(self) -> List[Dict]:
        """Extract foreign key relationships"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                tc.table_name,
                kcu.column_name,
                ccu.table_name AS foreign_table_name,
                ccu.column_name AS foreign_column_name,
                tc.constraint_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
        """)
        
        relationships = cursor.fetchall()
        cursor.close()
        return [dict(rel) for rel in relationships]
    
    async def _extract_indexes(self) -> Dict:
        """Extract database indexes"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                schemaname,
                tablename,
                indexname,
                indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
        """)
        
        indexes = cursor.fetchall()
        cursor.close()
        return {idx['indexname']: dict(idx) for idx in indexes}
    
    async def _extract_constraints(self) -> Dict:
        """Extract database constraints"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                tc.table_name,
                tc.constraint_name,
                tc.constraint_type,
                kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
                ON tc.constraint_name = kcu.constraint_name
            WHERE tc.table_schema = 'public'
        """)
        
        constraints = cursor.fetchall()
        cursor.close()
        return {const['constraint_name']: dict(const) for const in constraints}
    
    async def _extract_functions(self) -> List[Dict]:
        """Extract database functions and procedures"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                routine_name,
                routine_type,
                data_type,
                routine_definition
            FROM information_schema.routines
            WHERE routine_schema = 'public'
        """)
        
        functions = cursor.fetchall()
        cursor.close()
        return [dict(func) for func in functions]
    
    async def _extract_views(self) -> List[Dict]:
        """Extract database views"""
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT
                table_name,
                view_definition
            FROM information_schema.views
            WHERE table_schema = 'public'
        """)
        
        views = cursor.fetchall()
        cursor.close()
        return [dict(view) for view in views]
    
    async def _generate_semantic_analysis(self, schema_info: Dict) -> Dict:
        """Generate semantic analysis of the database schema"""
        tables = schema_info['tables']
        relationships = schema_info['relationships']
        
        # Analyze table relationships
        table_graph = {}
        for table_name in tables.keys():
            table_graph[table_name] = {
                'incoming': [],
                'outgoing': [],
                'centrality': 0
            }
        
        for rel in relationships:
            source_table = rel['table_name']
            target_table = rel['foreign_table_name']
            
            if source_table in table_graph:
                table_graph[source_table]['outgoing'].append(target_table)
            if target_table in table_graph:
                table_graph[target_table]['incoming'].append(source_table)
        
        # Calculate centrality scores
        for table_name, graph_info in table_graph.items():
            centrality = len(graph_info['incoming']) + len(graph_info['outgoing'])
            table_graph[table_name]['centrality'] = centrality
        
        # Identify core tables
        core_tables = sorted(table_graph.keys(), 
                           key=lambda x: table_graph[x]['centrality'], 
                           reverse=True)[:3]
        
        return {
            'table_graph': table_graph,
            'core_tables': core_tables,
            'database_complexity': len(tables),
            'relationship_density': len(relationships) / len(tables) if tables else 0
        }

class VectorStoreManager:
    """Manages vector storage for RAG system"""
    
    def __init__(self, mongodb_uri: str, collection_name: str = "bisee_vectors"):
        self.mongodb_uri = mongodb_uri
        self.collection_name = collection_name
        self.client = MongoClient(mongodb_uri)
        self.db = self.client.bisee_db
        self.collection = self.db[collection_name]
        self.sessions_collection = self.db.sessions
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB for fast similarity search
        self.chroma_client = chromadb.Client()
        self.chroma_collection = self.chroma_client.create_collection(
            name="bisee_insights",
            get_or_create=True
        )
    
    async def store_insight(self, insight: InsightResult, session_id: str):
        """Store insight in both MongoDB and ChromaDB"""
        # Prepare content for embedding
        content_text = json.dumps(insight.content)
        embedding = self.encoder.encode(content_text).tolist()
        
        # Store in MongoDB
        document = {
            'insight_id': insight.insight_id,
            'session_id': session_id,
            'insight_type': insight.insight_type,
            'content': insight.content,
            'metadata': insight.metadata,
            'confidence_score': insight.confidence_score,
            'timestamp': insight.timestamp,
            'embedding': embedding
        }
        
        await asyncio.to_thread(self.collection.insert_one, document)
        
        # Store in ChromaDB for fast retrieval
        self.chroma_collection.add(
            embeddings=[embedding],
            documents=[content_text],
            metadatas=[{
                'insight_id': insight.insight_id,
                'session_id': session_id,
                'insight_type': insight.insight_type,
                'confidence_score': insight.confidence_score
            }],
            ids=[insight.insight_id]
        )
    
    async def retrieve_similar_insights(self, query: str, session_id: str, limit: int = 5) -> List[Dict]:
        """Retrieve similar insights based on query"""
        query_embedding = self.encoder.encode(query).tolist()
        
        # Search in ChromaDB
        results = self.chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            where={"session_id": session_id}
        )
        
        # Get full documents from MongoDB
        insight_ids = results['ids'][0] if results['ids'] else []
        insights = []
        
        for insight_id in insight_ids:
            doc = await asyncio.to_thread(
                self.collection.find_one,
                {'insight_id': insight_id}
            )
            if doc:
                insights.append(doc)
        
        return insights
    
    async def get_session_insights(self, session_id: str) -> List[Dict]:
        """Get all insights for a session"""
        cursor = self.collection.find({'session_id': session_id})
        insights = await asyncio.to_thread(list, cursor)
        return insights
    
    async def store_session_context(self, session_context: SessionContext):
        """Store session context in MongoDB"""
        document = {
            'session_id': session_context.session_id,
            'user_id': session_context.user_id,
            'database_config': session_context.database_config,
            'conversation_history': session_context.conversation_history,
            'insights_generated': session_context.insights_generated,
            'current_context': session_context.current_context,
            'timestamp': session_context.timestamp
        }
        
        await asyncio.to_thread(
            self.sessions_collection.replace_one,
            {'session_id': session_context.session_id},
            document,
            upsert=True
        )
    
    async def get_session_context(self, session_id: str) -> Optional[SessionContext]:
        """Retrieve session context"""
        document = await asyncio.to_thread(
            self.sessions_collection.find_one,
            {'session_id': session_id}
        )
        
        if document:
            return SessionContext(
                session_id=document['session_id'],
                user_id=document['user_id'],
                database_config=document['database_config'],
                conversation_history=document['conversation_history'],
                insights_generated=document['insights_generated'],
                current_context=document['current_context'],
                timestamp=document['timestamp']
            )
        return None

class LLMManager:
    """Manages Groq LLM interactions"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama3-70b-8192"  # Use the larger model for better insights
    
    async def generate_response(self, prompt: str, system_prompt: str = None, max_tokens: int = 4000) -> str:
        """Generate response using Groq LLM"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return "I apologize, but I encountered an error generating the response."
    
    async def generate_json_response(self, prompt: str, system_prompt: str = None) -> Dict:
        """Generate structured JSON response"""
        response = await self.generate_response(prompt, system_prompt)
        
        try:
            # Extract JSON from response if wrapped in markdown
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            return json.loads(response)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON response: {response}")
            return {"error": "Failed to parse JSON response", "raw_response": response}

class DatabaseInsightAgent:
    """Agent specialized in database analysis and insights"""
    
    def __init__(self, llm_manager: LLMManager, schema_extractor: DatabaseSchemaExtractor):
        self.llm_manager = llm_manager
        self.schema_extractor = schema_extractor
        self.agent_type = AgentType.DATABASE_INSIGHT
    
    async def analyze_database_structure(self, context: Dict) -> Dict:
        """Analyze database structure and generate insights"""
        schema = await self.schema_extractor.extract_complete_schema()
        
        system_prompt = """You are a database analysis expert with deep learning capabilities. Analyze the provided database schema and generate comprehensive insights about the data structure, relationships, and business context. 

        Focus on:
        1. Data quality assessment and anomaly detection
        2. Business domain identification and entity relationships
        3. Key metrics and KPIs discovery
        4. Data lineage and dependencies
        5. Potential analysis opportunities and recommendations
        6. Performance optimization suggestions
        7. Business intelligence insights
        
        Use advanced pattern recognition to identify:
        - Hidden relationships between tables
        - Business processes reflected in the data structure
        - Potential data quality issues
        - Key performance indicators
        - Growth opportunities
        
        Return structured JSON output with detailed insights and confidence scores."""
        
        prompt = f"""
        Analyze this database schema and provide deep insights:
        
        Schema Information:
        {json.dumps(schema, indent=2, default=str)}
        
        Generate comprehensive insights about:
        1. Database structure complexity and health
        2. Business domain and operational patterns
        3. Data quality indicators and issues
        4. Key tables, relationships, and data flows
        5. Performance bottlenecks and optimization opportunities
        6. Business intelligence and analytics potential
        7. Strategic recommendations for data utilization
        
        Return JSON format:
        {{
            "database_health": {{
                "overall_score": 0.85,
                "complexity_level": "medium",
                "optimization_opportunities": []
            }},
            "business_insights": {{
                "primary_domain": "e-commerce",
                "key_processes": [],
                "growth_indicators": []
            }},
            "data_quality": {{
                "issues": [],
                "recommendations": []
            }},
            "analytics_opportunities": {{
                "kpis": [],
                "trending_analysis": [],
                "predictive_potential": []
            }},
            "strategic_recommendations": []
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'analysis_type': 'database_structure',
            'schema': schema,
            'insights': response,
            'metadata': {
                'tables_count': len(schema['tables']),
                'relationships_count': len(schema['relationships']),
                'complexity_score': schema['semantic_analysis']['database_complexity'],
                'analysis_timestamp': datetime.now().isoformat()
            }
        }
    
    async def analyze_table_data(self, table_name: str, context: Dict) -> Dict:
        """Analyze specific table data with deep learning insights"""
        table_info = context.get('schema', {}).get('tables', {}).get(table_name, {})
        
        system_prompt = """You are a data analyst expert with advanced pattern recognition capabilities. Analyze the provided table data and generate deep business insights using machine learning approaches.

        Focus on:
        1. Statistical analysis and distribution patterns
        2. Anomaly detection and outlier identification
        3. Correlation analysis and hidden relationships
        4. Trend identification and forecasting potential
        5. Business KPIs and performance metrics
        6. Data quality assessment and cleansing needs
        7. Actionable business recommendations
        
        Use advanced analytics to identify:
        - Seasonal patterns and trends
        - Customer segments and behaviors
        - Revenue optimization opportunities
        - Operational efficiency metrics
        - Risk indicators and early warning signs
        
        Provide structured analysis with confidence scores and actionable insights."""
        
        prompt = f"""
        Analyze this table data and provide deep insights:
        
        Table: {table_name}
        Information: {json.dumps(table_info, indent=2, default=str)}
        
        Generate comprehensive insights about:
        1. Data patterns, distributions, and statistical properties
        2. Business context and operational significance
        3. Quality assessment and data integrity
        4. Key findings, trends, and anomalies
        5. Performance metrics and KPIs
        6. Predictive analytics opportunities
        7. Strategic recommendations for business growth
        
        Return JSON format:
        {{
            "table_analysis": {{
                "data_quality_score": 0.92,
                "completeness": 0.98,
                "consistency": 0.95,
                "anomalies_detected": []
            }},
            "business_insights": {{
                "key_metrics": [],
                "trends": [],
                "patterns": [],
                "segments": []
            }},
            "statistical_analysis": {{
                "distributions": {{}},
                "correlations": [],
                "outliers": []
            }},
            "recommendations": {{
                "data_quality": [],
                "business_actions": [],
                "analytics_opportunities": []
            }},
            "confidence_score": 0.87
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'analysis_type': 'table_data',
            'table_name': table_name,
            'insights': response,
            'metadata': {
                'row_count': table_info.get('row_count', 0),
                'column_count': len(table_info.get('columns', [])),
                'business_domain': table_info.get('business_context', {}).get('primary_domain', 'unknown'),
                'analysis_timestamp': datetime.now().isoformat()
            }
        }

class SQLGeneratorAgent:
    """Agent specialized in SQL query generation with context awareness"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.SQL_GENERATOR
    
    async def generate_query(self, user_request: str, schema_context: Dict, session_context: Dict = None) -> Dict:
        """Generate SQL query based on user request with enhanced context"""
        system_prompt = """You are an expert SQL query generator with deep database knowledge. Generate optimized PostgreSQL queries based on user requests, database schema, and conversation context.

        Always:
        1. Use proper table and column names with double quotes
        2. Include comprehensive error handling
        3. Optimize for performance with proper indexing
        4. Add detailed comments explaining query logic
        5. Consider business context and user intent
        6. Provide query variations for different use cases
        7. Include data validation and quality checks
        
        Advanced capabilities:
        - Generate complex analytical queries
        - Create window functions for trend analysis
        - Build aggregation queries for KPIs
        - Construct joins across multiple tables
        - Implement filtering and segmentation
        - Generate time-series analysis queries
        
        Return structured JSON with query, explanation, and metadata."""
        
        schema_summary = self._create_enhanced_schema_summary(schema_context)
        context_info = self._extract_context_info(session_context) if session_context else {}
        
        # Continue from SQLGeneratorAgent class
        
        prompt = f"""
        Generate advanced SQL query for this request: "{user_request}"
        
        Database Schema:
        {schema_summary}
        
        Session Context:
        {json.dumps(context_info, indent=2, default=str)}
        
        Requirements:
        1. Use PostgreSQL advanced features
        2. Include proper error handling and data validation
        3. Optimize with appropriate indexes and joins
        4. Add comprehensive comments
        5. Consider business context and KPIs
        6. Include data quality checks
        7. Provide alternative query approaches
        
        Return JSON format:
        {{
            "query": {{
                "sql": "SELECT ...",
                "parameters": [],
                "query_type": "analytical",
                "estimated_performance": "high"
            }},
            "explanation": {{
                "business_context": "Analysis of customer purchase patterns",
                "query_logic": "Step by step explanation",
                "optimization_notes": "Index usage and performance considerations"
            }},
            "alternatives": [
                {{
                    "query": "Alternative SQL approach",
                    "use_case": "When to use this alternative",
                    "performance_impact": "performance comparison"
                }}
            ],
            "validation": {{
                "data_quality_checks": [],
                "expected_result_format": {{}},
                "potential_issues": []
            }},
            "confidence_score": 0.92
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'request': user_request,
            'generated_query': response,
            'schema_context': schema_context,
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'context_used': bool(session_context),
                'schema_tables_count': len(schema_context.get('tables', {}))
            }
        }
    
    def _create_enhanced_schema_summary(self, schema_context: Dict) -> str:
        """Create enhanced schema summary for LLM context"""
        tables = schema_context.get('tables', {})
        relationships = schema_context.get('relationships', [])
        
        summary = []
        summary.append("=== DATABASE SCHEMA SUMMARY ===\n")
        
        # Tables summary
        for table_name, table_info in tables.items():
            columns = table_info.get('columns', [])
            business_context = table_info.get('business_context', {})
            
            summary.append(f"Table: {table_name}")
            summary.append(f"- Business Domain: {business_context.get('primary_domain', 'unknown')}")
            summary.append(f"- Purpose: {business_context.get('table_purpose', 'unknown')}")
            summary.append(f"- Row Count: {table_info.get('row_count', 0)}")
            summary.append("- Key Columns:")
            
            for col in columns:
                col_info = f"  * {col['column_name']} ({col['data_type']})"
                if col.get('is_primary_key'):
                    col_info += " [PK]"
                if col.get('is_unique'):
                    col_info += " [UNIQUE]"
                summary.append(col_info)
            
            key_metrics = business_context.get('key_metrics', [])
            if key_metrics:
                summary.append(f"- Key Metrics: {', '.join(key_metrics)}")
            
            summary.append("")
        
        # Relationships summary
        if relationships:
            summary.append("=== RELATIONSHIPS ===")
            for rel in relationships:
                summary.append(f"- {rel['table_name']}.{rel['column_name']} -> {rel['foreign_table_name']}.{rel['foreign_column_name']}")
        
        return "\n".join(summary)
    
    def _extract_context_info(self, session_context: Dict) -> Dict:
        """Extract relevant context information"""
        return {
            'recent_queries': session_context.get('conversation_history', [])[-5:],
            'current_focus': session_context.get('current_context', {}),
            'user_preferences': session_context.get('user_preferences', {}),
            'previous_insights': session_context.get('insights_generated', [])[-3:]
        }

class VisualizationAgent:
    """Agent specialized in data visualization and reporting"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.VISUALIZATION
    
    async def generate_visualization(self, data_context: Dict, visualization_request: str, session_context: Dict = None) -> Dict:
        """Generate visualization recommendations and code"""
        system_prompt = """You are a data visualization expert with advanced knowledge of matplotlib, plotly, and seaborn. Generate comprehensive visualization solutions based on data context and user requirements.

        Capabilities:
        1. Create interactive dashboards with Plotly
        2. Generate statistical visualizations with matplotlib/seaborn
        3. Build responsive HTML/CSS visualizations
        4. Design business intelligence reports
        5. Create time-series and trend analysis charts
        6. Generate geographic and network visualizations
        7. Build custom visualization components
        
        Best practices:
        - Choose appropriate chart types for data patterns
        - Use color schemes that are accessible and meaningful
        - Include proper titles, labels, and legends
        - Implement interactive features when beneficial
        - Optimize for different screen sizes
        - Follow data visualization principles
        
        Return structured output with visualization code and configuration."""
        
        data_summary = self._create_data_summary(data_context)
        context_info = self._extract_visualization_context(session_context) if session_context else {}
        
        prompt = f"""
        Generate advanced visualization for: "{visualization_request}"
        
        Data Context:
        {data_summary}
        
        Session Context:
        {json.dumps(context_info, indent=2, default=str)}
        
        Requirements:
        1. Create multiple visualization options
        2. Include interactive features
        3. Generate both static and dynamic charts
        4. Provide HTML/CSS for web display
        5. Include business intelligence insights
        6. Add responsive design elements
        7. Generate summary statistics
        
        Return JSON format:
        {{
            "visualizations": [
                {{
                    "type": "plotly_interactive",
                    "title": "Interactive Revenue Trends",
                    "code": "python code here",
                    "html": "HTML embed code",
                    "config": {{}},
                    "insights": []
                }},
                {{
                    "type": "matplotlib_static",
                    "title": "Statistical Analysis",
                    "code": "python code here",
                    "insights": []
                }}
            ],
            "dashboard": {{
                "html": "Complete dashboard HTML",
                "css": "Styling code",
                "js": "Interactive JavaScript"
            }},
            "summary": {{
                "key_findings": [],
                "data_quality": {{}},
                "recommendations": []
            }},
            "metadata": {{
                "chart_types": [],
                "interactivity_level": "high",
                "responsive": true
            }}
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'visualization_request': visualization_request,
            'generated_visualizations': response,
            'data_context': data_context,
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'context_used': bool(session_context),
                'data_points': self._count_data_points(data_context)
            }
        }
    
    def _create_data_summary(self, data_context: Dict) -> str:
        """Create data summary for visualization context"""
        summary = []
        
        if 'query_results' in data_context:
            results = data_context['query_results']
            summary.append(f"Data Points: {len(results) if isinstance(results, list) else 'Unknown'}")
            
            if isinstance(results, list) and results:
                first_row = results[0]
                summary.append(f"Columns: {list(first_row.keys())}")
                summary.append(f"Data Types: {self._infer_data_types(first_row)}")
        
        if 'table_analysis' in data_context:
            analysis = data_context['table_analysis']
            summary.append(f"Table: {analysis.get('table_name', 'Unknown')}")
            summary.append(f"Business Domain: {analysis.get('business_domain', 'Unknown')}")
        
        return "\n".join(summary)
    
    def _infer_data_types(self, sample_row: Dict) -> Dict:
        """Infer data types from sample row"""
        type_map = {}
        for key, value in sample_row.items():
            if isinstance(value, (int, float)):
                type_map[key] = 'numeric'
            elif isinstance(value, str):
                type_map[key] = 'categorical'
            elif isinstance(value, datetime):
                type_map[key] = 'temporal'
            else:
                type_map[key] = 'unknown'
        return type_map
    
    def _extract_visualization_context(self, session_context: Dict) -> Dict:
        """Extract visualization-specific context"""
        return {
            'previous_charts': session_context.get('previous_visualizations', []),
            'user_preferences': session_context.get('visualization_preferences', {}),
            'dashboard_context': session_context.get('dashboard_state', {})
        }
    
    def _count_data_points(self, data_context: Dict) -> int:
        """Count data points in context"""
        if 'query_results' in data_context:
            results = data_context['query_results']
            return len(results) if isinstance(results, list) else 0
        return 0

class RecommendationAgent:
    """Agent specialized in business recommendations and strategic insights"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
        self.agent_type = AgentType.RECOMMENDATION
    
    async def generate_recommendations(self, user_input: str, context: Dict, session_context: Dict = None) -> Dict:
        """Generate business recommendations based on comprehensive analysis"""
        system_prompt = """You are a strategic business consultant with expertise in data-driven decision making. Generate actionable business recommendations based on database insights, user context, and industry best practices.

        Expertise areas:
        1. Business strategy and growth optimization
        2. Revenue enhancement and cost reduction
        3. Customer experience and retention
        4. Operational efficiency improvement
        5. Risk management and compliance
        6. Market analysis and competitive positioning
        7. Digital transformation and technology adoption
        
        Analysis approach:
        - Combine quantitative data insights with qualitative business understanding
        - Consider industry trends and market dynamics
        - Assess implementation feasibility and ROI
        - Provide timeline and resource requirements
        - Include risk assessment and mitigation strategies
        - Generate measurable KPIs and success metrics
        
        Return comprehensive recommendations with implementation roadmap."""
        
        context_summary = self._create_context_summary(context)
        business_context = self._extract_business_context(session_context) if session_context else {}
        
        prompt = f"""
        Generate strategic business recommendations for: "{user_input}"
        
        Data Context:
        {context_summary}
        
        Business Context:
        {json.dumps(business_context, indent=2, default=str)}
        
        Requirements:
        1. Provide actionable, data-driven recommendations
        2. Include implementation roadmap and timelines
        3. Assess business impact and ROI potential
        4. Consider resource requirements and constraints
        5. Include risk assessment and mitigation strategies
        6. Generate measurable KPIs and success metrics
        7. Provide short-term and long-term strategies
        
        Return JSON format:
        {{
            "executive_summary": {{
                "key_insights": [],
                "critical_actions": [],
                "expected_impact": "High/Medium/Low"
            }},
            "strategic_recommendations": [
                {{
                    "category": "Revenue Optimization",
                    "recommendation": "Detailed recommendation",
                    "rationale": "Data-driven justification",
                    "implementation": {{
                        "timeline": "3-6 months",
                        "resources": [],
                        "budget_estimate": "Low/Medium/High",
                        "dependencies": []
                    }},
                    "expected_outcomes": {{
                        "primary_metrics": [],
                        "secondary_benefits": [],
                        "roi_estimate": "percentage or value"
                    }},
                    "risk_assessment": {{
                        "risks": [],
                        "mitigation_strategies": [],
                        "probability": "Low/Medium/High"
                    }},
                    "priority": "High/Medium/Low"
                }}
            ],
            "implementation_roadmap": {{
                "phase_1": {{
                    "duration": "1-3 months",
                    "actions": [],
                    "milestones": []
                }},
                "phase_2": {{
                    "duration": "3-6 months",
                    "actions": [],
                    "milestones": []
                }},
                "phase_3": {{
                    "duration": "6-12 months",
                    "actions": [],
                    "milestones": []
                }}
            }},
            "success_metrics": {{
                "kpis": [],
                "measurement_frequency": "Monthly/Quarterly",
                "benchmarks": {}
            }},
            "next_steps": [],
            "confidence_score": 0.88
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        
        return {
            'agent_type': self.agent_type.value,
            'user_input': user_input,
            'recommendations': response,
            'context_used': context,
            'metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'context_depth': len(context),
                'session_context_used': bool(session_context)
            }
        }
    
    def _create_context_summary(self, context: Dict) -> str:
        """Create comprehensive context summary"""
        summary = []
        
        if 'database_insights' in context:
            insights = context['database_insights']
            summary.append("=== DATABASE INSIGHTS ===")
            summary.append(f"Business Domain: {insights.get('business_domain', 'Unknown')}")
            summary.append(f"Data Quality Score: {insights.get('data_quality_score', 'Unknown')}")
            summary.append(f"Key Metrics: {', '.join(insights.get('key_metrics', []))}")
        
        if 'query_insights' in context:
            query_insights = context['query_insights']
            summary.append("=== QUERY INSIGHTS ===")
            summary.append(f"Query Type: {query_insights.get('query_type', 'Unknown')}")
            summary.append(f"Data Points: {query_insights.get('data_points', 'Unknown')}")
        
        if 'visualization_insights' in context:
            viz_insights = context['visualization_insights']
            summary.append("=== VISUALIZATION INSIGHTS ===")
            summary.append(f"Chart Types: {', '.join(viz_insights.get('chart_types', []))}")
            summary.append(f"Key Findings: {', '.join(viz_insights.get('key_findings', []))}")
        
        return "\n".join(summary)
    
    def _extract_business_context(self, session_context: Dict) -> Dict:
        """Extract business-specific context"""
        return {
            'industry': session_context.get('industry', 'Unknown'),
            'company_size': session_context.get('company_size', 'Unknown'),
            'business_goals': session_context.get('business_goals', []),
            'current_challenges': session_context.get('challenges', []),
            'previous_recommendations': session_context.get('past_recommendations', [])
        }

class OrchestratorAgent:
    """Master orchestrator agent that coordinates all other agents"""
    
    def __init__(self, llm_manager: LLMManager, vector_store: VectorStoreManager):
        self.llm_manager = llm_manager
        self.vector_store = vector_store
        self.agent_type = AgentType.ORCHESTRATOR
        
        # Initialize specialized agents
        self.db_insight_agent = None
        self.sql_agent = None
        self.viz_agent = None
        self.recommendation_agent = None
        
    def setup_agents(self, schema_extractor: DatabaseSchemaExtractor):
        """Initialize all specialized agents"""
        self.db_insight_agent = DatabaseInsightAgent(self.llm_manager, schema_extractor)
        self.sql_agent = SQLGeneratorAgent(self.llm_manager)
        self.viz_agent = VisualizationAgent(self.llm_manager)
        self.recommendation_agent = RecommendationAgent(self.llm_manager)
    
    async def process_user_request(self, user_input: str, session_id: str, user_id: str) -> Dict:
        """Process user request through appropriate agent workflow"""
        # Analyze user intent
        intent = await self._analyze_user_intent(user_input)
        
        # Get session context
        session_context = await self.vector_store.get_session_context(session_id)
        if not session_context:
            session_context = SessionContext(
                session_id=session_id,
                user_id=user_id,
                database_config={},
                conversation_history=[],
                insights_generated=[],
                current_context={},
                timestamp=datetime.now()
            )
        
        # Route to appropriate agent workflow
        if intent['type'] == 'database_analysis':
            result = await self._handle_database_analysis(user_input, session_context, intent)
        elif intent['type'] == 'sql_query':
            result = await self._handle_sql_query(user_input, session_context, intent)
        elif intent['type'] == 'visualization':
            result = await self._handle_visualization(user_input, session_context, intent)
        elif intent['type'] == 'recommendation':
            result = await self._handle_recommendation(user_input, session_context, intent)
        elif intent['type'] == 'bisee_command':
            result = await self._handle_bisee_command(user_input, session_context, intent)
        else:
            result = await self._handle_general_query(user_input, session_context, intent)
        
        # Update session context
        session_context.conversation_history.append({
            'user_input': user_input,
            'response': result,
            'timestamp': datetime.now(),
            'intent': intent
        })
        
        # Store session context
        await self.vector_store.store_session_context(session_context)
        
        return result
    
    async def _analyze_user_intent(self, user_input: str) -> Dict:
        """Analyze user intent to determine appropriate agent workflow"""
        system_prompt = """You are an intent classification expert. Analyze user input and classify the intent to route to appropriate specialized agents.

        Intent categories:
        1. database_analysis - Questions about database structure, schema, data quality
        2. sql_query - Requests for data querying, filtering, aggregation
        3. visualization - Requests for charts, graphs, dashboards, reports
        4. recommendation - Requests for business insights, strategic advice
        5. bisee_command - Commands starting with /bisee for insight management
        6. general_query - General questions that may need multiple agents
        
        Return classification with confidence and routing information."""
        
        prompt = f"""
        Classify this user input: "{user_input}"
        
        Analyze:
        1. Primary intent category
        2. Specific requirements
        3. Agent routing strategy
        4. Context dependencies
        5. Expected output format
        
        Return JSON format:
        {{
            "type": "intent_category",
            "confidence": 0.95,
            "specific_request": "detailed description",
            "agents_needed": ["agent1", "agent2"],
            "expected_output": "description",
            "context_required": true,
            "bisee_command": null
        }}
        """
        
        response = await self.llm_manager.generate_json_response(prompt, system_prompt)
        return response
    
    async def _handle_database_analysis(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle database analysis requests"""
        # Get database insights
        db_insights = await self.db_insight_agent.analyze_database_structure(
            asdict(session_context)
        )
        
        # Generate insight result
        insight_result = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type='database_analysis',
            content=db_insights,
            metadata=intent,
            confidence_score=intent.get('confidence', 0.8),
            timestamp=datetime.now()
        )
        
        # Store in vector store
        await self.vector_store.store_insight(insight_result, session_context.session_id)
        
        # Update session context
        session_context.insights_generated.append(asdict(insight_result))
        
        return {
            'type': 'database_analysis',
            'insight_id': insight_result.insight_id,
            'content': db_insights,
            'summary': self._generate_insight_summary(db_insights),
            'recommendations': self._extract_recommendations(db_insights)
        }
    
    async def _handle_sql_query(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle SQL query generation and execution"""
        # Get schema context
        schema_context = session_context.current_context.get('schema', {})
        
        # Generate SQL query
        sql_result = await self.sql_agent.generate_query(
            user_input, 
            schema_context, 
            asdict(session_context)
        )
        
        # Execute query (if safe)
        query_results = await self._execute_safe_query(sql_result)
        
        # Generate insight result
        insight_result = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type='sql_query',
            content={
                'sql_generation': sql_result,
                'query_results': query_results
            },
            metadata=intent,
            confidence_score=intent.get('confidence', 0.8),
            timestamp=datetime.now()
        )
        
        # Store in vector store
        await self.vector_store.store_insight(insight_result, session_context.session_id)
        
        # Update session context
        session_context.insights_generated.append(asdict(insight_result))
        
        return {
            'type': 'sql_query',
            'insight_id': insight_result.insight_id,
            'sql_query': sql_result['generated_query']['query']['sql'],
            'results': query_results,
            'explanation': sql_result['generated_query']['explanation'],
            'summary': self._generate_query_summary(query_results)
        }
    
    async def _handle_visualization(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle visualization requests"""
        # Get relevant data context
        data_context = await self._get_visualization_data_context(session_context)
        
        # Generate visualization
        viz_result = await self.viz_agent.generate_visualization(
            data_context,
            user_input,
            asdict(session_context)
        )
        
        # Generate insight result
        insight_result = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type='visualization',
            content=viz_result,
            metadata=intent,
            confidence_score=intent.get('confidence', 0.8),
            timestamp=datetime.now()
        )
        
        # Store in vector store
        await self.vector_store.store_insight(insight_result, session_context.session_id)
        
        # Update session context
        session_context.insights_generated.append(asdict(insight_result))
        
        return {
            'type': 'visualization',
            'insight_id': insight_result.insight_id,
            'visualizations': viz_result['generated_visualizations']['visualizations'],
            'dashboard': viz_result['generated_visualizations']['dashboard'],
            'summary': viz_result['generated_visualizations']['summary']
        }
    
    async def _handle_recommendation(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle business recommendation requests"""
        # Gather comprehensive context
        context = await self._gather_comprehensive_context(session_context)
        
        # Generate recommendations
        rec_result = await self.recommendation_agent.generate_recommendations(
            user_input,
            context,
            asdict(session_context)
        )
        
        # Generate insight result
        insight_result = InsightResult(
            insight_id=str(uuid.uuid4()),
            insight_type='recommendation',
            content=rec_result,
            metadata=intent,
            confidence_score=intent.get('confidence', 0.8),
            timestamp=datetime.now()
        )
        
        # Store in vector store
        await self.vector_store.store_insight(insight_result, session_context.session_id)
        
        # Update session context
        session_context.insights_generated.append(asdict(insight_result))
        
        return {
            'type': 'recommendation',
            'insight_id': insight_result.insight_id,
            'recommendations': rec_result['recommendations'],
            'executive_summary': rec_result['recommendations']['executive_summary'],
            'implementation_roadmap': rec_result['recommendations']['implementation_roadmap']
        }
    
    async def _handle_bisee_command(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle /bisee commands for insight management"""
        command = intent.get('bisee_command', '')
        
        if command.startswith('/bisee list'):
            # List all insights for the session
            insights = await self.vector_store.get_session_insights(session_context.session_id)
            return {
                'type': 'bisee_list',
                'insights': [
                    {
                        'insight_id': insight['insight_id'],
                        'type': insight['insight_type'],
                        'timestamp': insight['timestamp'],
                        'summary': self._generate_insight_summary(insight['content'])
                    }
                    for insight in insights
                ]
            }
        
        elif command.startswith('/bisee summarize'):
            # Summarize selected insights
            insight_ids = self._extract_insight_ids(command)
            return await self._summarize_insights(insight_ids, session_context)
        
        elif command.startswith('/bisee visualize'):
            # Visualize selected insights
            insight_ids = self._extract_insight_ids(command)
            return await self._visualize_insights(insight_ids, session_context)
        
        else:
            return {
                'type': 'bisee_help',
                'commands': [
                    '/bisee list - List all generated insights',
                    '/bisee summarize [insight_ids] - Summarize selected insights',
                    '/bisee visualize [insight_ids] - Visualize selected insights'
                ]
            }
    
    async def _handle_general_query(self, user_input: str, session_context: SessionContext, intent: Dict) -> Dict:
        """Handle general queries that may need multiple agents"""
        # This is a complex orchestration that may involve multiple agents
        # For now, route to recommendation agent as it can handle general business queries
        return await self._handle_recommendation(user_input, session_context, intent)
    
    async def _execute_safe_query(self, sql_result: Dict) -> List[Dict]:
        """Execute SQL query safely with proper validation"""
        # This would include query validation, execution limits, etc.
        # For now, return mock results
        return [{'message': 'Query execution not implemented in this example'}]
    
    async def _get_visualization_data_context(self, session_context: SessionContext) -> Dict:
        """Get data context for visualization"""
        # Get recent SQL results or database insights
        recent_insights = session_context.insights_generated[-3:]
        data_context = {}
        
        for insight in recent_insights:
            if insight['insight_type'] == 'sql_query':
                data_context['query_results'] = insight['content'].get('query_results', [])
            elif insight['insight_type'] == 'database_analysis':
                data_context['table_analysis'] = insight['content']
        
        return data_context
    
    async def _gather_comprehensive_context(self, session_context: SessionContext) -> Dict:
        """Gather comprehensive context for recommendations"""
        context = {}
        
        # Get database insights
        db_insights = [i for i in session_context.insights_generated if i['insight_type'] == 'database_analysis']
        if db_insights:
            context['database_insights'] = db_insights[-1]['content']
        
        # Get query insights
        query_insights = [i for i in session_context.insights_generated if i['insight_type'] == 'sql_query']
        if query_insights:
            context['query_insights'] = query_insights[-1]['content']
        
        # Get visualization insights
        viz_insights = [i for i in session_context.insights_generated if i['insight_type'] == 'visualization']
        if viz_insights:
            context['visualization_insights'] = viz_insights[-1]['content']
        
        return context
    
    def _generate_insight_summary(self, insight_content: Dict) -> str:
        """Generate a summary of insight content"""
        if 'insights' in insight_content:
            insights = insight_content['insights']
            if 'business_insights' in insights:
                return f"Business domain: {insights['business_insights'].get('primary_domain', 'Unknown')}"
        return "Generated insight available"
    
    def _generate_query_summary(self, query_results: List[Dict]) -> str:
        """Generate a summary of query results"""
        if query_results:
            return f"Query returned {len(query_results)} rows"
        return "Query executed successfully"
    
    def _extract_recommendations(self, db_insights: Dict) -> List[str]:
        """Extract recommendations from database insights"""
        insights = db_insights.get('insights', {})
        recommendations = []
        
        if 'strategic_recommendations' in insights:
            recommendations = insights['strategic_recommendations']
        elif 'recommendations' in insights:
            recommendations = insights['recommendations']
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def _extract_insight_ids(self, command: str) -> List[str]:
        """Extract insight IDs from bisee command"""
        # Simple implementation - would need more sophisticated parsing
        parts = command.split()
        return [part for part in parts if part.startswith('insight_')]
    
    async def _summarize_insights(self, insight_ids: List[str], session_context: SessionContext) -> Dict:
        """Summarize selected insights"""
        # Get insights from vector store
        insights = []
        for insight_id in insight_ids:
            insight = next((i for i in session_context.insights_generated if i['insight_id'] == insight_id), None)
            if insight:
                insights.append(insight)
        
        # Generate summary using LLM
        summary_prompt = f"""
        Summarize these insights:
        {json.dumps(insights, indent=2, default=str)}
        
        Provide a comprehensive summary highlighting key findings and recommendations.
        """
        
        summary = await self.llm_manager.generate