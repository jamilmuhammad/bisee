#!/usr/bin/env python3
"""
Test script for PostgreSQL database connector functionality
"""

from app import BusinessIntelligenceAgent
import os

def test_sqlite():
    """Test SQLite functionality"""
    print("🧪 Testing SQLite functionality...")
    
    try:
        agent = BusinessIntelligenceAgent(db_type="sqlite", db_path="test_sqlite.db")
        
        if agent.initialize():
            tables = agent.get_available_tables()
            print(f"✅ SQLite tables created: {tables}")
            
            if tables:
                schema = agent.db_connector.get_schema_info(tables[0])
                print(f"✅ SQLite schema retrieved: {len(schema['columns'])} columns")
            
            agent.db_connector.close()
            
            # Clean up test file
            if os.path.exists("test_sqlite.db"):
                os.remove("test_sqlite.db")
                
            return True
        else:
            print("❌ SQLite initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ SQLite test failed: {e}")
        return False

def test_postgresql_config():
    """Test PostgreSQL configuration (without actual connection)"""
    print("🧪 Testing PostgreSQL configuration...")
    
    try:
        # Test configuration creation
        agent = BusinessIntelligenceAgent(
            db_type="postgresql",
            host="localhost",
            port=5432,
            database="test_db",
            user="test_user",
            password="test_pass"
        )
        
        print("✅ PostgreSQL configuration successful")
        
        # Test connection parameters
        expected_params = {
            "host": "localhost",
            "port": 5432,
            "database": "test_db",
            "user": "test_user",
            "password": "test_pass"
        }
        
        if agent.db_connector.connection_params == expected_params:
            print("✅ PostgreSQL parameters correctly set")
            return True
        else:
            print("❌ PostgreSQL parameters incorrect")
            return False
            
    except Exception as e:
        print(f"❌ PostgreSQL configuration test failed: {e}")
        return False

def test_database_types():
    """Test database type validation"""
    print("🧪 Testing database type validation...")
    
    try:
        # Test invalid database type
        try:
            agent = BusinessIntelligenceAgent(db_type="invalid_db")
            print("❌ Should have failed with invalid database type")
            return False
        except ValueError:
            print("✅ Invalid database type correctly rejected")
        
        # Test case insensitivity
        agent_upper = BusinessIntelligenceAgent(db_type="SQLITE")
        agent_mixed = BusinessIntelligenceAgent(db_type="PostgreSQL")
        
        if (agent_upper.db_connector.db_type == "sqlite" and 
            agent_mixed.db_connector.db_type == "postgresql"):
            print("✅ Database type case handling works")
            return True
        else:
            print("❌ Database type case handling failed")
            return False
            
    except Exception as e:
        print(f"❌ Database type validation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running PostgreSQL Database Connector Tests")
    print("=" * 50)
    
    tests = [
        test_sqlite,
        test_postgresql_config,
        test_database_types
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("📊 Test Results")
    print("-" * 20)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
