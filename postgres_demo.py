#!/usr/bin/env python3
"""
Final verification and demonstration of PostgreSQL BI Agent
"""

from app import BusinessIntelligenceAgent
import pandas as pd

def demonstrate_postgresql_bi():
    """Demonstrate the full PostgreSQL BI capabilities"""
    print("🎯 PostgreSQL Business Intelligence Agent Demo")
    print("Database: db_bisee")
    print("=" * 50)
    
    # Your PostgreSQL configuration
    agent = BusinessIntelligenceAgent(
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="db_bisee",
        user="postgres",
        password="password"
    )
    
    if not agent.db_connector.connect():
        print("❌ Failed to connect")
        return
    
    # Get tables
    tables = agent.get_available_tables()
    print(f"📊 Tables: {tables}")
    
    # Focus on the main table
    main_table = "TransactionData"
    
    print(f"\n🔍 Detailed Analysis of {main_table}")
    print("-" * 40)
    
    # 1. Schema Information
    schema = agent.db_connector.get_schema_info(main_table)
    print(f"📋 Schema Info:")
    print(f"  • Table: {schema['table_name']}")
    print(f"  • Columns: {len(schema['columns'])}")
    print(f"  • Rows: {schema['row_count']}")
    print(f"  • Primary Keys: {schema['primary_keys']}")
    
    print(f"\n📊 Column Details:")
    for col in schema['columns'][:5]:  # Show first 5 columns
        print(f"  • {col['name']}: {col['type']} {'(PK)' if col['primary_key'] else ''}")
    
    # 2. Raw Data Query
    print(f"\n📈 Sample Data (Raw Query):")
    raw_data = agent.db_connector.execute_query(f"SELECT * FROM {main_table} LIMIT 3")
    if not raw_data.empty:
        print(raw_data.to_string(index=False))
    
    # 3. Business Analysis
    print(f"\n💼 Business Analysis:")
    analysis = agent.analyzer.analyze_table(main_table)
    
    # Data Quality
    quality = analysis['data_quality']
    print(f"📊 Data Quality:")
    print(f"  • Total Records: {quality['total_rows']}")
    print(f"  • Columns: {quality['total_columns']}")
    print(f"  • Missing Values: {sum(quality['missing_values'].values())}")
    print(f"  • Duplicates: {quality['duplicate_rows']}")
    
    # Insights
    print(f"\n💡 Key Insights:")
    for i, insight in enumerate(analysis['insights'][:3], 1):
        print(f"  {i}. {insight}")
    
    # Recommendations
    print(f"\n🎯 Recommendations:")
    for rec in analysis['recommendations'][:2]:
        print(f"  • {rec['action']} ({rec['priority']})")
        print(f"    {rec['details']}")
    
    # 4. SQL Queries Generated
    print(f"\n🔍 Generated SQL Queries:")
    
    # Categorical analysis
    categorical_query = agent.query_generator.generate_categorical_analysis(main_table, "transactiontype")
    print(f"📊 Transaction Type Analysis:")
    cat_result = agent.db_connector.execute_query(categorical_query)
    if not cat_result.empty:
        print(cat_result.to_string(index=False))
    
    # Numeric analysis
    numeric_query = agent.query_generator.generate_numeric_analysis(main_table, "amount")
    print(f"\n💰 Amount Analysis:")
    num_result = agent.db_connector.execute_query(numeric_query)
    if not num_result.empty:
        print(num_result.to_string(index=False))
    
    # Close connection
    agent.db_connector.close()
    print(f"\n✅ Demo Complete! PostgreSQL integration successful.")

if __name__ == "__main__":
    demonstrate_postgresql_bi()
