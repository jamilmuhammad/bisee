#!/usr/bin/env python3
"""
Test script for your PostgreSQL database (db_bisee)
"""

from app import BusinessIntelligenceAgent
import sys

def test_your_postgresql_db():
    """Test connection to your PostgreSQL database"""
    print("🐘 Testing Your PostgreSQL Database: db_bisee")
    print("=" * 50)
    
    # Your database configuration
    db_config = {
        "db_type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "db_bisee",
        "user": "postgres",
        "password": "password"
    }
    
    try:
        # Initialize the agent with your PostgreSQL config
        print("🔌 Connecting to PostgreSQL database...")
        agent = BusinessIntelligenceAgent(**db_config)
        
        if not agent.initialize():
            print("❌ Failed to initialize PostgreSQL agent")
            print("Make sure PostgreSQL is running and accessible")
            return False
        
        print("✅ Successfully connected to db_bisee!")
        
        # Get available tables
        tables = agent.get_available_tables()
        print(f"📊 Available tables in db_bisee: {tables}")
        
        if not tables:
            print("ℹ️  No existing tables found. Sample data will be created.")
        
        # Test table operations
        print("\n🔍 Testing database operations...")
        
        # Check if our sample table exists, if not it was created during initialization
        tables_after_init = agent.get_available_tables()
        print(f"📊 Tables after initialization: {tables_after_init}")
        
        if tables_after_init:
            # Analyze the first table
            table_name = tables_after_init[0]
            print(f"\n📋 Analyzing table: {table_name}")
            
            # Get schema info
            schema = agent.db_connector.get_schema_info(table_name)
            print(f"✅ Schema retrieved - {len(schema['columns'])} columns, {schema['row_count']} rows")
            
            # Get sample data
            sample_data = agent.db_connector.get_sample_data(table_name, 3)
            print(f"✅ Sample data retrieved - {len(sample_data)} rows")
            print("Sample data preview:")
            print(sample_data.head())
            
            # Generate analysis report
            print(f"\n📊 Generating business report for {table_name}...")
            report = agent.analyze_table(table_name)
            
            if 'insights' in report:
                print("✅ Business insights generated:")
                for i, insight in enumerate(report['insights'][:3], 1):
                    print(f"  {i}. {insight}")
            
            if 'recommendations' in report:
                print("✅ Recommendations generated:")
                for rec in report['recommendations'][:2]:
                    print(f"  • {rec['action']} ({rec['priority']} priority)")
        
        # Test connection close
        agent.db_connector.close()
        print("\n✅ Database connection closed successfully")
        
        print("\n🎉 All PostgreSQL tests passed!")
        print("Your db_bisee database is working perfectly with the BI Agent!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Install PostgreSQL support: pip install psycopg2-binary")
        return False
    except Exception as e:
        print(f"❌ Error connecting to PostgreSQL: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure PostgreSQL is running: brew services start postgresql")
        print("2. Check if the database exists: psql -U postgres -c '\\l'")
        print("3. Verify credentials and database name")
        print("4. Check if PostgreSQL is listening on localhost:5432")
        return False

def show_connection_info():
    """Display connection information"""
    print("\n📝 Connection Details:")
    print("-" * 30)
    print("Host: localhost")
    print("Port: 5432") 
    print("Database: db_bisee")
    print("User: postgres")
    print("Password: [hidden]")
    print()

if __name__ == "__main__":
    show_connection_info()
    
    success = test_your_postgresql_db()
    
    if success:
        print("\n🚀 Ready for production use!")
        print("You can now use your PostgreSQL database with the BI Agent.")
    else:
        print("\n🔧 Please check the connection and try again.")
    
    sys.exit(0 if success else 1)
