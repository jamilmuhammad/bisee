import sqlite3
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
    print("⚠️  PostgreSQL support not available. Install psycopg2 to use PostgreSQL: pip install psycopg2-binary")

class DatabaseConnector:
    """Handles database connections and schema analysis for SQLite and PostgreSQL"""
    
    def __init__(self, db_type: str = "sqlite", db_path: str = None, **kwargs):
        """
        Initialize database connector
        
        Args:
            db_type: Database type - 'sqlite' or 'postgresql'
            db_path: For SQLite - path to database file
            **kwargs: For PostgreSQL - host, port, database, user, password
        """
        self.db_type = db_type.lower()
        self.connection = None
        self.schema_info = {}
        
        if self.db_type == "sqlite":
            self.db_path = db_path or "business_data.db"
            self.connection_params = {"db_path": self.db_path}
        elif self.db_type == "postgresql":
            if not POSTGRESQL_AVAILABLE:
                raise ImportError("PostgreSQL support requires psycopg2. Install with: pip install psycopg2-binary")
            
            self.connection_params = {
                "host": kwargs.get("host", "localhost"),
                "port": kwargs.get("port", 5432),
                "database": kwargs.get("database", "business_data"),
                "user": kwargs.get("user", "postgres"),
                "password": kwargs.get("password", "")
            }
        else:
            raise ValueError(f"Unsupported database type: {db_type}. Use 'sqlite' or 'postgresql'")
        
    def connect(self):
        """Establish database connection"""
        try:
            if self.db_type == "sqlite":
                self.connection = sqlite3.connect(self.connection_params["db_path"])
                self.connection.row_factory = sqlite3.Row
                print(f"✅ Connected to SQLite database: {self.connection_params['db_path']}")
            elif self.db_type == "postgresql":
                self.connection = psycopg2.connect(
                    host=self.connection_params["host"],
                    port=self.connection_params["port"],
                    database=self.connection_params["database"],
                    user=self.connection_params["user"],
                    password=self.connection_params["password"],
                    cursor_factory=RealDictCursor
                )
                print(f"✅ Connected to PostgreSQL database: {self.connection_params['database']} at {self.connection_params['host']}")
            
            return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            if self.db_type == "postgresql":
                self.connection.close()
            elif self.db_type == "sqlite":
                self.connection.close()
            self.connection = None
            print("✅ Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def get_tables(self) -> List[str]:
        """Get all tables in the database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        
        if self.db_type == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
        elif self.db_type == "postgresql":
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
            """)
            tables = [row['table_name'] for row in cursor.fetchall()]
        
        cursor.close()
        return tables
    
    def get_schema_info(self, table_name: str) -> Dict:
        """Get detailed schema information for a table"""
        if not self.connection:
            return {}
        
        cursor = self.connection.cursor()
        
        schema_info = {
            'table_name': table_name,
            'columns': [],
            'primary_keys': [],
            'foreign_keys': [],
            'row_count': 0
        }
        
        if self.db_type == "sqlite":
            # SQLite schema query
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            
            for col in columns:
                schema_info['columns'].append({
                    'name': col[1],
                    'type': col[2],
                    'not_null': bool(col[3]),
                    'default': col[4],
                    'primary_key': bool(col[5])
                })
                
                if col[5]:  # Primary key
                    schema_info['primary_keys'].append(col[1])
            
            # Get foreign keys for SQLite
            cursor.execute(f"PRAGMA foreign_key_list({table_name})")
            fks = cursor.fetchall()
            for fk in fks:
                schema_info['foreign_keys'].append({
                    'column': fk[3],
                    'references_table': fk[2],
                    'references_column': fk[4]
                })
                
        elif self.db_type == "postgresql":
            # PostgreSQL schema query
            cursor.execute("""
                SELECT 
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    c.column_default,
                    CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END as is_primary_key
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.constraint_type = 'PRIMARY KEY'
                        AND tc.table_name = %s
                        AND tc.table_schema = 'public'
                ) pk ON c.column_name = pk.column_name
                WHERE c.table_name = %s 
                    AND c.table_schema = 'public'
                ORDER BY c.ordinal_position;
            """, (table_name, table_name))
            columns = cursor.fetchall()
            
            for col in columns:
                schema_info['columns'].append({
                    'name': col['column_name'],
                    'type': col['data_type'],
                    'not_null': col['is_nullable'] == 'NO',
                    'default': col['column_default'],
                    'primary_key': col['is_primary_key']
                })
                
                if col['is_primary_key']:  # Primary key
                    schema_info['primary_keys'].append(col['column_name'])
            
            # Get foreign keys for PostgreSQL
            cursor.execute("""
                SELECT
                    kcu.column_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                    ON tc.constraint_name = kcu.constraint_name
                    AND tc.table_schema = kcu.table_schema
                JOIN information_schema.constraint_column_usage AS ccu
                    ON ccu.constraint_name = tc.constraint_name
                    AND ccu.table_schema = tc.table_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                    AND tc.table_name = %s
                    AND tc.table_schema = 'public';
            """, (table_name,))
            fks = cursor.fetchall()
            for fk in fks:
                schema_info['foreign_keys'].append({
                    'column': fk['column_name'],
                    'references_table': fk['foreign_table_name'],
                    'references_column': fk['foreign_column_name']
                })
        
        # Get row count (works for both databases)
        if self.db_type == "postgresql":
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        else:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = cursor.fetchone()
        if self.db_type == "postgresql":
            schema_info['row_count'] = result['count']
        else:
            schema_info['row_count'] = result[0]
        
        cursor.close()
        return schema_info
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        if not self.connection:
            return pd.DataFrame()
        
        try:
            if self.db_type == "postgresql":
                # For pandas operations, use a regular connection without RealDictCursor
                import psycopg2
                temp_conn = psycopg2.connect(
                    host=self.connection_params["host"],
                    port=self.connection_params["port"],
                    database=self.connection_params["database"],
                    user=self.connection_params["user"],
                    password=self.connection_params["password"]
                )
                result = pd.read_sql_query(query, temp_conn)
                temp_conn.close()
                return result
            else:
                return pd.read_sql_query(query, self.connection)
        except Exception as e:
            print(f"❌ Query execution failed: {e}")
            return pd.DataFrame()
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from table"""
        if self.db_type == "postgresql":
            # Use quoted table name for PostgreSQL
            query = f'SELECT * FROM "{table_name}" LIMIT {limit}'
        else:
            query = f"SELECT * FROM {table_name} LIMIT {limit}"
        return self.execute_query(query)

class DataAnalyzer:
    """Analyzes data and generates business insights"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
        self.insights = {}
        
    def analyze_table(self, table_name: str) -> Dict:
        """Comprehensive table analysis"""
        schema = self.db.get_schema_info(table_name)
        sample_data = self.db.get_sample_data(table_name, 100)
        
        if sample_data.empty:
            return {"error": "No data available for analysis"}
        
        analysis = {
            'table_name': table_name,
            'schema': schema,
            'data_quality': self._analyze_data_quality(sample_data),
            'statistical_summary': self._generate_statistical_summary(sample_data),
            'patterns': self._identify_patterns(sample_data),
            'insights': self._generate_insights(sample_data, schema),
            'recommendations': self._generate_recommendations(sample_data, schema)
        }
        
        # Add comprehensive text summary
        analysis['text_summary'] = self._generate_text_summary(analysis, table_name)
        
        return analysis
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict:
        """Analyze data quality metrics"""
        quality_metrics = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        # Convert numpy types to Python types for JSON serialization
        for col, dtype in quality_metrics['data_types'].items():
            quality_metrics['data_types'][col] = str(dtype)
        
        return quality_metrics
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict:
        """Generate statistical summary"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        summary = {
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns analysis
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        for col in categorical_cols:
            summary['categorical_summary'][col] = {
                'unique_count': df[col].nunique(),
                'top_values': df[col].value_counts().head(5).to_dict()
            }
        
        return summary
    
    def _identify_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify data patterns and trends"""
        patterns = {
            'correlations': {},
            'outliers': {},
            'trends': {}
        }
        
        # Correlation analysis for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            patterns['correlations'] = {
                'strong_positive': [],
                'strong_negative': []
            }
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        pair = f"{corr_matrix.columns[i]} vs {corr_matrix.columns[j]}"
                        if corr_val > 0:
                            patterns['correlations']['strong_positive'].append({
                                'pair': pair,
                                'correlation': float(corr_val)
                            })
                        else:
                            patterns['correlations']['strong_negative'].append({
                                'pair': pair,
                                'correlation': float(corr_val)
                            })
        
        return patterns
    
    def _generate_insights(self, df: pd.DataFrame, schema: Dict) -> List[str]:
        """Generate business insights"""
        insights = []
        
        # Data volume insight
        row_count = len(df)
        col_count = len(df.columns)
        insights.append(f"Dataset contains {row_count} records across {col_count} dimensions")
        
        # Missing data insight
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 10]
        if len(high_missing) > 0:
            insights.append(f"Data quality concern: {len(high_missing)} columns have >10% missing values")
        else:
            insights.append("Excellent data completeness with minimal missing values")
        
        # Business-specific insights for transaction data
        if 'transactiontype' in df.columns or 'TransactionType' in df.columns:
            type_col = 'transactiontype' if 'transactiontype' in df.columns else 'TransactionType'
            type_dist = df[type_col].value_counts()
            most_common = type_dist.index[0] if len(type_dist) > 0 else 'Unknown'
            insights.append(f"Most frequent transaction type: '{most_common}' ({type_dist.iloc[0]} occurrences)")
            
            if len(type_dist) > 1:
                insights.append(f"Transaction diversity: {len(type_dist)} different transaction types identified")
        
        # Amount analysis if present
        amount_cols = [col for col in df.columns if 'amount' in col.lower()]
        if amount_cols:
            amount_col = amount_cols[0]
            if df[amount_col].dtype in ['int64', 'float64']:
                avg_amount = df[amount_col].mean()
                max_amount = df[amount_col].max()
                min_amount = df[amount_col].min()
                insights.append(f"Transaction amounts range from ${min_amount:.2f} to ${max_amount:.2f} (avg: ${avg_amount:.2f})")
                
                # High value transaction insight
                high_value_threshold = df[amount_col].quantile(0.9)
                high_value_count = (df[amount_col] > high_value_threshold).sum()
                if high_value_count > 0:
                    insights.append(f"High-value transactions (>${high_value_threshold:.2f}+): {high_value_count} detected")
        
        # Status analysis
        status_cols = [col for col in df.columns if 'status' in col.lower()]
        if status_cols:
            status_col = status_cols[0]
            status_dist = df[status_col].value_counts()
            if 'Success' in status_dist.index or 'success' in status_dist.index:
                success_key = 'Success' if 'Success' in status_dist.index else 'success'
                success_rate = (status_dist.get(success_key, 0) / len(df)) * 100
                insights.append(f"Transaction success rate: {success_rate:.1f}%")
                
                if success_rate < 90:
                    insights.append("Below-optimal success rate detected - investigate failed transactions")
        
        # Fraud analysis
        fraud_cols = [col for col in df.columns if 'fraud' in col.lower()]
        if fraud_cols:
            fraud_col = fraud_cols[0]
            if df[fraud_col].dtype == 'object':
                fraud_count = (df[fraud_col].str.lower() == 'true').sum()
                fraud_rate = (fraud_count / len(df)) * 100
                if fraud_rate > 0:
                    insights.append(f"Fraud detection: {fraud_count} suspicious transactions identified ({fraud_rate:.1f}% rate)")
                else:
                    insights.append("No fraudulent transactions detected in current dataset")
        
        # Device analysis
        device_cols = [col for col in df.columns if 'device' in col.lower()]
        if device_cols:
            device_col = device_cols[0]
            device_dist = df[device_col].value_counts()
            most_used_device = device_dist.index[0] if len(device_dist) > 0 else 'Unknown'
            insights.append(f"Primary transaction device: '{most_used_device}' ({device_dist.iloc[0]} transactions)")
        
        # Performance insights
        latency_cols = [col for col in df.columns if 'latency' in col.lower()]
        if latency_cols:
            latency_col = latency_cols[0]
            if df[latency_col].dtype in ['int64', 'float64']:
                avg_latency = df[latency_col].mean()
                max_latency = df[latency_col].max()
                if avg_latency > 100:
                    insights.append(f"Performance concern: Average latency {avg_latency:.1f}ms (max: {max_latency:.1f}ms)")
                else:
                    insights.append(f"Good performance: Average latency {avg_latency:.1f}ms")
        
        # Categorical distribution insights
        for col in df.select_dtypes(include=['object']).columns:
            if col.lower() not in ['transactiontype', 'status', 'fraud', 'device']:  # Skip already analyzed
                unique_pct = df[col].nunique() / len(df) * 100
                if unique_pct < 5:  # Low cardinality
                    top_value = df[col].value_counts().iloc[0]
                    top_pct = top_value / len(df) * 100
                    insights.append(f"'{col}' shows high concentration: {top_pct:.1f}% in dominant category")
        
        # Numeric insights
        for col in df.select_dtypes(include=[np.number]).columns:
            if col.lower() not in ['amount', 'latency']:  # Skip already analyzed
                if df[col].std() > 0:
                    cv = df[col].std() / df[col].mean() * 100 if df[col].mean() != 0 else 0
                    if cv > 100:
                        insights.append(f"'{col}' shows high variability (CV: {cv:.1f}%) - investigate outliers")
        
        # Time-based insights if timestamp available
        timestamp_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if timestamp_cols:
            insights.append("Temporal data available - time-series analysis recommended")
        
        return insights[:10]  # Limit to top 10 most relevant insights
    
    def _generate_recommendations(self, df: pd.DataFrame, schema: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            recommendations.append({
                'category': 'Data Quality',
                'priority': 'High',
                'action': 'Implement data validation and cleansing procedures',
                'details': f"Address missing values in {missing_data[missing_data > 0].to_dict()}",
                'timeline': 'Immediate'
            })
        
        # Performance recommendations
        if len(df) > 1000:
            recommendations.append({
                'category': 'Performance',
                'priority': 'Medium',
                'action': 'Consider data indexing and query optimization',
                'details': 'Large dataset detected - optimize database queries for better performance',
                'timeline': 'Within 2 weeks'
            })
        
        # Analysis recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 2:
            recommendations.append({
                'category': 'Analysis',
                'priority': 'Medium',
                'action': 'Conduct deeper statistical analysis',
                'details': 'Multiple numeric variables available for advanced analytics',
                'timeline': 'Within 1 week'
            })
        
        return recommendations
    
    def _generate_text_summary(self, analysis: Dict, table_name: str) -> str:
        """Generate comprehensive text summary of the analysis"""
        summary_parts = []
        
        # Header
        summary_parts.append(f"📊 BUSINESS INTELLIGENCE SUMMARY FOR {table_name.upper()}")
        summary_parts.append("=" * 60)
        
        # Executive Overview
        schema = analysis.get('schema', {})
        quality = analysis.get('data_quality', {})
        
        summary_parts.append("\n🎯 EXECUTIVE OVERVIEW:")
        summary_parts.append(f"• Dataset contains {schema.get('row_count', 0)} transaction records")
        summary_parts.append(f"• Data spans {len(schema.get('columns', []))} dimensions")
        summary_parts.append(f"• Overall data quality score: {self._calculate_data_quality_score(analysis)}/100")
        
        # Data Quality Assessment
        summary_parts.append("\n📋 DATA QUALITY ASSESSMENT:")
        if quality.get('missing_values'):
            missing_total = sum(quality['missing_values'].values())
            if missing_total == 0:
                summary_parts.append("• ✅ No missing values detected - excellent data completeness")
            else:
                missing_cols = [k for k, v in quality['missing_values'].items() if v > 0]
                summary_parts.append(f"• ⚠️  Missing values found in {len(missing_cols)} columns: {missing_cols}")
        
        if quality.get('duplicate_rows', 0) > 0:
            summary_parts.append(f"• ⚠️  {quality['duplicate_rows']} duplicate records detected")
        else:
            summary_parts.append("• ✅ No duplicate records found")
        
        # Business Insights
        insights = analysis.get('insights', [])
        if insights:
            summary_parts.append("\n💡 KEY BUSINESS INSIGHTS:")
            for i, insight in enumerate(insights, 1):
                summary_parts.append(f"• {insight}")
        
        # Statistical Highlights
        stats = analysis.get('statistical_summary', {})
        if stats.get('numeric_summary'):
            summary_parts.append("\n📈 STATISTICAL HIGHLIGHTS:")
            for col, stat_data in list(stats['numeric_summary'].items())[:3]:  # Top 3 numeric columns
                if isinstance(stat_data, dict) and 'mean' in stat_data:
                    mean_val = stat_data['mean']
                    std_val = stat_data.get('std', 0)
                    summary_parts.append(f"• {col}: Average ${mean_val:.2f}, Variability {std_val:.2f}")
        
        # Categorical Analysis
        if stats.get('categorical_summary'):
            summary_parts.append("\n🏷️  CATEGORICAL ANALYSIS:")
            for col, cat_data in list(stats['categorical_summary'].items())[:2]:  # Top 2 categorical columns
                if isinstance(cat_data, dict) and 'top_values' in cat_data:
                    top_category = list(cat_data['top_values'].keys())[0] if cat_data['top_values'] else 'N/A'
                    unique_count = cat_data.get('unique_count', 0)
                    summary_parts.append(f"• {col}: {unique_count} unique values, most common: '{top_category}'")
        
        # Actionable Recommendations
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            summary_parts.append("\n🎯 ACTIONABLE RECOMMENDATIONS:")
            high_priority = [r for r in recommendations if r.get('priority') == 'High']
            medium_priority = [r for r in recommendations if r.get('priority') == 'Medium']
            
            if high_priority:
                summary_parts.append("  HIGH PRIORITY:")
                for rec in high_priority:
                    summary_parts.append(f"  • {rec.get('action', 'N/A')} - {rec.get('timeline', 'Immediate')}")
            
            if medium_priority:
                summary_parts.append("  MEDIUM PRIORITY:")
                for rec in medium_priority:
                    summary_parts.append(f"  • {rec.get('action', 'N/A')} - {rec.get('timeline', 'Within 2 weeks')}")
        
        # Performance Indicators
        summary_parts.append("\n📊 PERFORMANCE INDICATORS:")
        if schema.get('row_count', 0) > 1000:
            summary_parts.append("• 🔥 Large dataset - consider indexing optimization")
        elif schema.get('row_count', 0) > 100:
            summary_parts.append("• ✅ Medium dataset - good for analysis")
        else:
            summary_parts.append("• 📝 Small dataset - suitable for detailed examination")
        
        # Patterns and Trends
        patterns = analysis.get('patterns', {})
        if patterns.get('correlations'):
            strong_pos = patterns['correlations'].get('strong_positive', [])
            strong_neg = patterns['correlations'].get('strong_negative', [])
            if strong_pos or strong_neg:
                summary_parts.append("\n🔗 CORRELATION PATTERNS:")
                for corr in strong_pos[:2]:  # Top 2 positive correlations
                    summary_parts.append(f"• Strong positive: {corr.get('pair', 'N/A')} ({corr.get('correlation', 0):.2f})")
                for corr in strong_neg[:2]:  # Top 2 negative correlations
                    summary_parts.append(f"• Strong negative: {corr.get('pair', 'N/A')} ({corr.get('correlation', 0):.2f})")
        
        # Conclusion
        summary_parts.append("\n🎉 CONCLUSION:")
        quality_score = self._calculate_data_quality_score(analysis)
        if quality_score >= 90:
            summary_parts.append("• Dataset exhibits excellent quality and is ready for advanced analytics")
        elif quality_score >= 70:
            summary_parts.append("• Dataset shows good quality with minor improvements needed")
        else:
            summary_parts.append("• Dataset requires data quality improvements before analysis")
        
        summary_parts.append("• Regular monitoring and analysis recommended for optimal business insights")
        summary_parts.append("\n" + "=" * 60)
        
        return "\n".join(summary_parts)
    
    def _calculate_data_quality_score(self, analysis: Dict) -> float:
        """Calculate data quality score (0-100)"""
        quality_metrics = analysis.get('data_quality', {})
        
        # Calculate completeness score
        total_cells = quality_metrics.get('total_rows', 0) * quality_metrics.get('total_columns', 0)
        missing_cells = sum(quality_metrics.get('missing_values', {}).values())
        completeness = (total_cells - missing_cells) / total_cells if total_cells > 0 else 0
        
        # Calculate uniqueness score
        total_rows = quality_metrics.get('total_rows', 1)
        duplicate_rows = quality_metrics.get('duplicate_rows', 0)
        uniqueness = 1 - (duplicate_rows / total_rows) if total_rows > 0 else 1
        
        # Overall score (simple average, can be made more sophisticated)
        overall_score = (completeness + uniqueness) / 2 * 100
        
        return round(overall_score, 2)

class QueryGenerator:
    """Generates optimized SQL queries for analysis"""
    
    def __init__(self, db_connector: DatabaseConnector):
        self.db = db_connector
    
    def _format_table_name(self, table_name: str) -> str:
        """Format table name based on database type"""
        if self.db.db_type == "postgresql":
            return f'"{table_name}"'
        else:
            return table_name
    
    def _format_column_name(self, column_name: str) -> str:
        """Format column name based on database type"""
        if self.db.db_type == "postgresql":
            return f'"{column_name}"'
        else:
            return f'"{column_name}"'  # Use quotes for both for consistency
    
    def generate_summary_query(self, table_name: str) -> str:
        """Generate basic summary query"""
        formatted_table = self._format_table_name(table_name)
        return f"""
        -- Basic Summary Query for {table_name}
        SELECT 
            COUNT(*) as total_records
        FROM {formatted_table};
        """
    
    def generate_categorical_analysis(self, table_name: str, column_name: str) -> str:
        """Generate categorical analysis query"""
        formatted_table = self._format_table_name(table_name)
        formatted_column = self._format_column_name(column_name)
        return f"""
        -- Categorical Analysis for {column_name}
        SELECT 
            {formatted_column},
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM {formatted_table}), 2) as percentage
        FROM {formatted_table}
        WHERE {formatted_column} IS NOT NULL
        GROUP BY {formatted_column}
        ORDER BY count DESC;
        """
    
    def generate_numeric_analysis(self, table_name: str, column_name: str) -> str:
        """Generate numeric analysis query"""
        formatted_table = self._format_table_name(table_name)
        formatted_column = self._format_column_name(column_name)
        return f"""
        -- Numeric Analysis for {column_name}
        SELECT 
            COUNT(*) as count,
            MIN({formatted_column}) as min_value,
            MAX({formatted_column}) as max_value,
            AVG({formatted_column}) as avg_value,
            ROUND(AVG({formatted_column}), 2) as avg_rounded
        FROM {formatted_table}
        WHERE {formatted_column} IS NOT NULL;
        """
    
    def generate_time_series_analysis(self, table_name: str, date_column: str, value_column: str) -> str:
        """Generate time series analysis query"""
        formatted_table = self._format_table_name(table_name)
        formatted_date = self._format_column_name(date_column)
        formatted_value = self._format_column_name(value_column)
        
        if self.db.db_type == "postgresql":
            date_func = f"DATE({formatted_date})"
        else:
            date_func = f"DATE({formatted_date})"
        
        return f"""
        -- Time Series Analysis
        SELECT 
            {date_func} as date,
            COUNT(*) as transaction_count,
            AVG({formatted_value}) as avg_value
        FROM {formatted_table}
        WHERE {formatted_date} IS NOT NULL
        GROUP BY {date_func}
        ORDER BY date;
        """

class VisualizationGenerator:
    """Generates matplotlib visualizations based on data analysis"""
    
    def __init__(self):
        self.viz_config = {}
        plt.style.use('seaborn-v0_8')
        
    def generate_viz_config(self, data: pd.DataFrame, analysis_type: str) -> Dict:
        """Generate visualization configuration JSON"""
        config = {
            'chart_type': self._determine_chart_type(data, analysis_type),
            'data_format': self._format_data_for_viz(data),
            'styling': self._get_styling_config(),
            'interactivity': self._get_interactivity_config(),
            'layout': self._get_layout_config(data)
        }
        return config
    
    def _determine_chart_type(self, data: pd.DataFrame, analysis_type: str) -> str:
        """Determine optimal chart type based on data characteristics"""
        if analysis_type == 'categorical':
            if len(data) <= 10:
                return 'bar_chart'
            else:
                return 'horizontal_bar_chart'
        elif analysis_type == 'numeric':
            return 'histogram'
        elif analysis_type == 'time_series':
            return 'line_chart'
        elif analysis_type == 'correlation':
            return 'heatmap'
        else:
            return 'scatter_plot'
    
    def _format_data_for_viz(self, data: pd.DataFrame) -> Dict:
        """Format data for visualization"""
        return {
            'x_values': data.iloc[:, 0].tolist() if len(data.columns) > 0 else [],
            'y_values': data.iloc[:, 1].tolist() if len(data.columns) > 1 else [],
            'labels': data.columns.tolist(),
            'data_points': len(data)
        }
    
    def _get_styling_config(self) -> Dict:
        """Get styling configuration"""
        return {
            'color_palette': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'],
            'figure_size': [12, 8],
            'title_size': 16,
            'label_size': 12,
            'grid': True,
            'legend': True
        }
    
    def _get_interactivity_config(self) -> Dict:
        """Get interactivity configuration"""
        return {
            'hover_tooltips': True,
            'zoom_enabled': True,
            'pan_enabled': True,
            'selection_enabled': False
        }
    
    def _get_layout_config(self, data: pd.DataFrame) -> Dict:
        """Get layout configuration"""
        return {
            'title': f'Data Analysis Visualization',
            'x_axis_title': data.columns[0] if len(data.columns) > 0 else 'X Axis',
            'y_axis_title': data.columns[1] if len(data.columns) > 1 else 'Y Axis',
            'show_values': True,
            'rotation': 45 if len(data) > 10 else 0
        }
    
    def create_visualization(self, data: pd.DataFrame, config: Dict) -> plt.Figure:
        """Create actual matplotlib visualization"""
        fig, ax = plt.subplots(figsize=config['styling']['figure_size'])
        
        chart_type = config['chart_type']
        x_values = config['data_format']['x_values']
        y_values = config['data_format']['y_values']
        colors = config['styling']['color_palette']
        
        if chart_type == 'bar_chart':
            bars = ax.bar(x_values, y_values, color=colors[0])
            ax.set_xlabel(config['layout']['x_axis_title'])
            ax.set_ylabel(config['layout']['y_axis_title'])
            
            # Add value labels on bars
            if config['layout']['show_values']:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}', ha='center', va='bottom')
        
        elif chart_type == 'line_chart':
            ax.plot(x_values, y_values, marker='o', color=colors[0], linewidth=2)
            ax.set_xlabel(config['layout']['x_axis_title'])
            ax.set_ylabel(config['layout']['y_axis_title'])
        
        elif chart_type == 'histogram':
            ax.hist(y_values, bins=20, color=colors[0], alpha=0.7, edgecolor='black')
            ax.set_xlabel(config['layout']['x_axis_title'])
            ax.set_ylabel('Frequency')
        
        # Apply common styling
        ax.set_title(config['layout']['title'], fontsize=config['styling']['title_size'])
        ax.grid(config['styling']['grid'], alpha=0.3)
        
        if config['layout']['rotation'] > 0:
            plt.xticks(rotation=config['layout']['rotation'])
        
        plt.tight_layout()
        return fig

class BusinessIntelligenceAgent:
    """Main AI Agent for Business Intelligence"""
    
    def __init__(self, db_type: str = "sqlite", db_path: str = None, **kwargs):
        """
        Initialize Business Intelligence Agent
        
        Args:
            db_type: Database type - 'sqlite' or 'postgresql'
            db_path: For SQLite - path to database file
            **kwargs: For PostgreSQL - host, port, database, user, password
        """
        self.db_connector = DatabaseConnector(db_type, db_path, **kwargs)
        self.analyzer = DataAnalyzer(self.db_connector)
        self.query_generator = QueryGenerator(self.db_connector)
        self.viz_generator = VisualizationGenerator()
        
    def initialize(self) -> bool:
        """Initialize the agent and database connection"""
        success = self.db_connector.connect()
        if success:
            self._create_sample_data()
        return success
    
    def _create_sample_data(self):
        """Create sample transaction data for demonstration"""
        cursor = self.db_connector.connection.cursor()
        
        if self.db_connector.db_type == "sqlite":
            # SQLite table creation
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS TransactionData (
                TransactionID INTEGER PRIMARY KEY,
                TransactionType TEXT,
                Amount REAL,
                TransactionStatus TEXT,
                FraudFlag TEXT,
                Timestamp TEXT,
                DeviceUsed TEXT,
                GeolocationLatLong TEXT,
                LatencyMs INTEGER,
                SliceBandwidthMbps REAL
            )
            ''')
            
            # Insert sample data for SQLite
            sample_data = [
                (1, 'Transfer', 1500.00, 'Success', 'False', '2024-01-01 10:00:00', 'Mobile', '40.7128,-74.0060', 45, 50.5),
                (2, 'Deposit', 2000.00, 'Success', 'False', '2024-01-01 11:00:00', 'Desktop', '34.0522,-118.2437', 32, 75.2),
                (3, 'Withdrawal', 500.00, 'Failed', 'True', '2024-01-01 12:00:00', 'Mobile', '41.8781,-87.6298', 156, 25.1),
                (4, 'Transfer', 750.00, 'Success', 'False', '2024-01-01 13:00:00', 'Tablet', '29.7604,-95.3698', 67, 45.8),
                (5, 'Deposit', 3000.00, 'Success', 'True', '2024-01-01 14:00:00', 'Desktop', '40.7128,-74.0060', 234, 15.3),
            ]
            
            cursor.executemany('''
            INSERT OR REPLACE INTO TransactionData VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', sample_data)
            
        elif self.db_connector.db_type == "postgresql":
            # First, clear any existing data that might be corrupted
            cursor.execute('DROP TABLE IF EXISTS "TransactionData" CASCADE;')
            
            # PostgreSQL table creation with proper case handling
            cursor.execute('''
            CREATE TABLE "TransactionData" (
                "TransactionID" SERIAL PRIMARY KEY,
                "TransactionType" VARCHAR(50),
                "Amount" DECIMAL(10,2),
                "TransactionStatus" VARCHAR(20),
                "FraudFlag" VARCHAR(5),
                "Timestamp" TIMESTAMP,
                "DeviceUsed" VARCHAR(20),
                "GeolocationLatLong" VARCHAR(50),
                "LatencyMs" INTEGER,
                "SliceBandwidthMbps" DECIMAL(5,1)
            )
            ''')
            
            # Clear any existing data first
            cursor.execute('DELETE FROM "TransactionData"')
            
            # Insert fresh sample data for PostgreSQL
            sample_data = [
                (1, 'Transfer', 1500.00, 'Success', 'False', '2024-01-01 10:00:00', 'Mobile', '40.7128,-74.0060', 45, 50.5),
                (2, 'Deposit', 2000.00, 'Success', 'False', '2024-01-01 11:00:00', 'Desktop', '34.0522,-118.2437', 32, 75.2),
                (3, 'Withdrawal', 500.00, 'Failed', 'True', '2024-01-01 12:00:00', 'Mobile', '41.8781,-87.6298', 156, 25.1),
                (4, 'Transfer', 750.00, 'Success', 'False', '2024-01-01 13:00:00', 'Tablet', '29.7604,-95.3698', 67, 45.8),
                (5, 'Deposit', 3000.00, 'Success', 'True', '2024-01-01 14:00:00', 'Desktop', '40.7128,-74.0060', 234, 15.3),
                (6, 'Payment', 250.00, 'Success', 'False', '2024-01-01 15:00:00', 'Mobile', '37.7749,-122.4194', 78, 42.3),
                (7, 'Transfer', 800.00, 'Failed', 'True', '2024-01-01 16:00:00', 'Desktop', '40.7128,-74.0060', 298, 18.7),
                (8, 'Deposit', 1200.00, 'Success', 'False', '2024-01-01 17:00:00', 'Tablet', '34.0522,-118.2437', 56, 65.1),
                (9, 'Withdrawal', 300.00, 'Success', 'False', '2024-01-01 18:00:00', 'Mobile', '41.8781,-87.6298', 89, 38.9),
                (10, 'Payment', 450.00, 'Success', 'False', '2024-01-01 19:00:00', 'Desktop', '29.7604,-95.3698', 34, 72.4),
            ]
            
            for data in sample_data:
                cursor.execute('''
                INSERT INTO "TransactionData" ("TransactionID", "TransactionType", "Amount", "TransactionStatus", 
                                              "FraudFlag", "Timestamp", "DeviceUsed", "GeolocationLatLong", 
                                              "LatencyMs", "SliceBandwidthMbps") 
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ("TransactionID") DO UPDATE SET
                    "TransactionType" = EXCLUDED."TransactionType",
                    "Amount" = EXCLUDED."Amount",
                    "TransactionStatus" = EXCLUDED."TransactionStatus",
                    "FraudFlag" = EXCLUDED."FraudFlag",
                    "Timestamp" = EXCLUDED."Timestamp",
                    "DeviceUsed" = EXCLUDED."DeviceUsed",
                    "GeolocationLatLong" = EXCLUDED."GeolocationLatLong",
                    "LatencyMs" = EXCLUDED."LatencyMs",
                    "SliceBandwidthMbps" = EXCLUDED."SliceBandwidthMbps"
                ''', data)
        
        self.db_connector.connection.commit()
        cursor.close()
        print("✅ Sample data created successfully")
    
    def get_available_tables(self) -> List[str]:
        """Get list of available tables for analysis"""
        return self.db_connector.get_tables()
    
    def analyze_table(self, table_name: str) -> Dict:
        """Perform comprehensive table analysis"""
        print(f"🔍 Analyzing table: {table_name}")
        analysis = self.analyzer.analyze_table(table_name)
        return analysis
    
    def generate_business_report(self, table_name: str) -> Dict:
        """Generate comprehensive business report"""
        analysis = self.analyze_table(table_name)
        
        if 'error' in analysis:
            return analysis
        
        # Generate SQL queries for different analysis types
        queries = {
            'summary': self.query_generator.generate_summary_query(table_name),
            'categorical_queries': {},
            'numeric_queries': {}
        }
        
        # Get actual data for visualization
        data = self.db_connector.get_sample_data(table_name, 100)
        
        # Generate visualizations
        visualizations = []
        
        # Categorical analysis
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            query = self.query_generator.generate_categorical_analysis(table_name, col)
            queries['categorical_queries'][col] = query
            
            # Generate visualization
            viz_data = self.db_connector.execute_query(query)
            if not viz_data.empty:
                viz_config = self.viz_generator.generate_viz_config(viz_data, 'categorical')
                viz_config['title'] = f'{col} Distribution'
                visualizations.append({
                    'type': 'categorical',
                    'column': col,
                    'config': viz_config
                })
        
        # Numeric analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            query = self.query_generator.generate_numeric_analysis(table_name, col)
            queries['numeric_queries'][col] = query
            
            # Generate visualization
            viz_data = data[[col]].dropna()
            if not viz_data.empty:
                viz_config = self.viz_generator.generate_viz_config(viz_data, 'numeric')
                viz_config['title'] = f'{col} Distribution'
                visualizations.append({
                    'type': 'numeric',
                    'column': col,
                    'config': viz_config
                })
        
        # Compile final report
        report = {
            'executive_summary': self._generate_executive_summary(analysis),
            'data_insights': analysis['insights'],
            'sql_queries': queries,
            'visualizations': visualizations,
            'recommendations': analysis['recommendations'],
            'raw_analysis': analysis
        }
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict) -> Dict:
        """Generate executive summary from analysis"""
        schema = analysis['schema']
        insights = analysis['insights']
        
        summary = {
            'total_records': schema['row_count'],
            'data_dimensions': len(schema['columns']),
            'key_findings': insights[:3],  # Top 3 insights
            'data_quality_score': self.analyzer._calculate_data_quality_score(analysis),
            'business_impact': 'Medium to High'  # This could be more sophisticated
        }
        
        return summary
    
    def create_visualization(self, viz_config: Dict) -> plt.Figure:
        """Create matplotlib visualization from config"""
        # Create sample data based on config
        data_format = viz_config['data_format']
        sample_data = pd.DataFrame({
            'x': data_format['x_values'],
            'y': data_format['y_values'] if data_format['y_values'] else data_format['x_values']
        })
        
        return self.viz_generator.create_visualization(sample_data, viz_config)

def main():
    """Main application entry point"""
    print("🚀 Business Intelligence AI Agent System")
    print("=" * 50)
    
    # Example 1: SQLite (default)
    # print("\n📊 Example 1: Using SQLite Database")
    # agent_sqlite = BusinessIntelligenceAgent(db_type="sqlite", db_path="business_data.db")
    
    # if not agent_sqlite.initialize():
    #     print("❌ Failed to initialize SQLite agent")
    #     return
    
    # Example 2: PostgreSQL (uncomment and configure as needed)
    print("\n🐘 Example 2: Using PostgreSQL Database")
    agent_postgres = BusinessIntelligenceAgent(
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="db_bisee",
        user="postgres",
        password="password"
    )
    
    if not agent_postgres.initialize():
        print("❌ Failed to initialize PostgreSQL agent")
    else:
        postgres_tables = agent_postgres.get_available_tables()
        print(f"📊 PostgreSQL tables: {postgres_tables}")
    
    # Continue with SQLite example
    # agent = agent_sqlite

    agent = agent_postgres
    
    # Get available tables
    tables = agent.get_available_tables()
    print(f"📊 Available tables: {tables}")
    
    # Analyze the first table (or user can select)
    if tables:
        table_name = tables[0]  # You can modify this to allow user selection
        print(f"\n🔍 Analyzing table: {table_name}")
        
        # Generate comprehensive business report
        report = agent.generate_business_report(table_name)
        
        # Display results
        print("\n📋 EXECUTIVE SUMMARY")
        print("-" * 30)
        summary = report['executive_summary']
        print(f"Total Records: {summary['total_records']}")
        print(f"Data Quality Score: {summary['data_quality_score']}%")
        print(f"Key Findings:")
        for i, finding in enumerate(summary['key_findings'], 1):
            print(f"  {i}. {finding}")
        
        print("\n💡 RECOMMENDATIONS")
        print("-" * 30)
        for rec in report['recommendations']:
            print(f"• {rec['action']} ({rec['priority']} priority)")
            print(f"  {rec['details']}")
        
        print("\n📊 VISUALIZATION CONFIGS")
        print("-" * 30)
        for viz in report['visualizations']:
            print(f"• {viz['type'].title()} chart for {viz['column']}")
            print(f"  Config: {json.dumps(viz['config'], indent=2)}")
        
        # Create and show a sample visualization
        if report['visualizations']:
            print("\n🎨 Creating sample visualization...")
            first_viz = report['visualizations'][0]
            fig = agent.create_visualization(first_viz['config'])
            plt.show()
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()