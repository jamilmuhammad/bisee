#!/usr/bin/env python3
"""
PostgreSQL Database Connector Example

This script demonstrates how to use the BusinessIntelligenceAgent with PostgreSQL.
Make sure you have PostgreSQL running and the required dependencies installed.

Installation:
    pip install -r requirements.txt

PostgreSQL Setup:
    1. Install PostgreSQL
    2. Create a database: CREATE DATABASE business_data;
    3. Update the connection parameters below
"""

from app import BusinessIntelligenceAgent

def main():
    """Demonstrate PostgreSQL connectivity"""
    print("🐘 PostgreSQL Business Intelligence Example")
    print("=" * 50)
    
    # PostgreSQL connection parameters
    # Update these with your actual PostgreSQL credentials
    pg_config = {
        "db_type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "business_data",
        "user": "postgres",
        "password": "your_password_here"  # Update this!
    }
    
    try:
        # Initialize the agent with PostgreSQL
        agent = BusinessIntelligenceAgent(**pg_config)
        
        if not agent.initialize():
            print("❌ Failed to initialize PostgreSQL agent")
            print("Make sure PostgreSQL is running and credentials are correct")
            return
        
        # Get available tables
        tables = agent.get_available_tables()
        print(f"📊 Available PostgreSQL tables: {tables}")
        
        # Analyze the first table if available
        if tables:
            table_name = tables[0]
            print(f"\n🔍 Analyzing table: {table_name}")
            
            # Generate comprehensive business report
            report = agent.generate_business_report(table_name)
            
            # Display results
            if 'executive_summary' in report:
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
            else:
                print("❌ Failed to generate report")
        
        # Close the connection
        agent.db_connector.close()
        print("\n✅ PostgreSQL analysis complete!")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install PostgreSQL support: pip install psycopg2-binary")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check your PostgreSQL connection parameters and ensure the database is running")

if __name__ == "__main__":
    main()
