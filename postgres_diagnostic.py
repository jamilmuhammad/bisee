#!/usr/bin/env python3
"""
PostgreSQL Connection Diagnostic Script
"""

import psycopg2
from psycopg2.extras import RealDictCursor
import sys

def test_direct_connection():
    """Test direct PostgreSQL connection without our wrapper"""
    print("🔍 Testing Direct PostgreSQL Connection")
    print("=" * 40)
    
    connection_params = {
        "host": "localhost",
        "port": 5432,
        "database": "db_bisee", 
        "user": "postgres",
        "password": "password"
    }
    
    try:
        print("🔌 Attempting direct connection...")
        conn = psycopg2.connect(**connection_params)
        print("✅ Direct connection successful!")
        
        # Test basic query
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f"📊 PostgreSQL Version: {version[0]}")
        
        # List existing tables
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE';
        """)
        tables = cursor.fetchall()
        print(f"📋 Existing tables: {[t[0] for t in tables]}")
        
        cursor.close()
        conn.close()
        print("✅ Direct connection test passed!")
        return True
        
    except psycopg2.OperationalError as e:
        print(f"❌ Connection failed: {e}")
        print("\nPossible issues:")
        print("1. PostgreSQL server is not running")
        print("2. Database 'db_bisee' doesn't exist")
        print("3. Wrong credentials")
        print("4. PostgreSQL not listening on localhost:5432")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_database_exists():
    """Test if the database exists"""
    print("\n🔍 Checking if database exists")
    print("=" * 30)
    
    try:
        # Connect to postgres database to check if db_bisee exists
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="postgres",  # Connect to default postgres db
            user="postgres",
            password="password"
        )
        
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'db_bisee';")
        exists = cursor.fetchone()
        
        if exists:
            print("✅ Database 'db_bisee' exists")
        else:
            print("❌ Database 'db_bisee' does not exist")
            print("💡 Creating database...")
            
            # Create the database
            conn.autocommit = True
            cursor.execute("CREATE DATABASE db_bisee;")
            print("✅ Database 'db_bisee' created successfully!")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error checking/creating database: {e}")
        return False

def test_with_our_agent():
    """Test with our BusinessIntelligenceAgent"""
    print("\n🔍 Testing with BusinessIntelligenceAgent")
    print("=" * 35)
    
    try:
        from app import BusinessIntelligenceAgent
        
        agent = BusinessIntelligenceAgent(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="db_bisee",
            user="postgres",
            password="password"
        )
        
        print("🔌 Connecting with BI Agent...")
        if agent.db_connector.connect():
            print("✅ BI Agent connection successful!")
            
            # Test getting tables
            tables = agent.get_available_tables()
            print(f"📊 Tables found: {tables}")
            
            agent.db_connector.close()
            return True
        else:
            print("❌ BI Agent connection failed")
            return False
            
    except Exception as e:
        print(f"❌ BI Agent test failed: {e}")
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 PostgreSQL Diagnostic Tool")
    print("Testing connection to db_bisee database")
    print("=" * 50)
    
    # Test 1: Direct connection
    if not test_direct_connection():
        # Test 2: Check if database exists and create if needed
        if test_database_exists():
            # Retry direct connection
            print("\n🔄 Retrying direct connection...")
            if not test_direct_connection():
                print("❌ Still cannot connect after creating database")
                return False
        else:
            return False
    
    # Test 3: Test with our agent
    if not test_with_our_agent():
        return False
    
    print("\n🎉 All tests passed!")
    print("Your PostgreSQL database is ready to use!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
