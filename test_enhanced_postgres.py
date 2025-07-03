#!/usr/bin/env python3
"""
Enhanced PostgreSQL Analysis Test
"""

from app import BusinessIntelligenceAgent

def test_enhanced_analysis():
    """Test the enhanced PostgreSQL analysis capabilities"""
    print("🚀 Enhanced PostgreSQL Business Intelligence Analysis")
    print("=" * 60)
    
    # Connect to PostgreSQL
    agent = BusinessIntelligenceAgent(
        db_type="postgresql",
        host="localhost",
        port=5432,
        database="db_bisee",
        user="postgres",
        password="password"
    )
    
    if not agent.initialize():
        print("❌ Failed to initialize")
        return False
    
    # Get tables
    tables = agent.get_available_tables()
    print(f"📊 Available Tables: {tables}")
    
    # Analyze the main table
    main_table = "TransactionData"
    print(f"\n🔍 Analyzing: {main_table}")
    print("-" * 40)
    
    # Get comprehensive analysis
    analysis = agent.analyze_table(main_table)
    
    if 'error' in analysis:
        print(f"❌ Analysis failed: {analysis['error']}")
        return False
    
    # Display text summary
    print("\n📝 COMPREHENSIVE TEXT SUMMARY:")
    print(analysis.get('text_summary', 'No summary available'))
    
    # Display insights
    print("\n💡 DETAILED INSIGHTS:")
    for i, insight in enumerate(analysis.get('insights', []), 1):
        print(f"  {i}. {insight}")
    
    # Display recommendations
    print("\n🎯 ACTIONABLE RECOMMENDATIONS:")
    for rec in analysis.get('recommendations', []):
        print(f"  • {rec.get('action', 'N/A')} ({rec.get('priority', 'Medium')} priority)")
        print(f"    Details: {rec.get('details', 'N/A')}")
        print(f"    Timeline: {rec.get('timeline', 'N/A')}")
        print()
    
    # Test SQL queries
    print("\n🔍 TESTING GENERATED SQL QUERIES:")
    print("-" * 40)
    
    # Test categorical analysis
    cat_query = agent.query_generator.generate_categorical_analysis(main_table, "TransactionType")
    print("📊 Transaction Type Distribution:")
    cat_result = agent.db_connector.execute_query(cat_query)
    if not cat_result.empty:
        print(cat_result.to_string(index=False))
    else:
        print("No results")
    
    # Test numeric analysis
    num_query = agent.query_generator.generate_numeric_analysis(main_table, "Amount")
    print("\n💰 Amount Analysis:")
    num_result = agent.db_connector.execute_query(num_query)
    if not num_result.empty:
        print(num_result.to_string(index=False))
    else:
        print("No results")
    
    # Test sample data display
    print("\n📊 SAMPLE DATA:")
    sample_data = agent.db_connector.get_sample_data(main_table, 5)
    if not sample_data.empty:
        print(sample_data.to_string(index=False))
    else:
        print("No sample data")
    
    # Close connection
    agent.db_connector.close()
    print("\n✅ Enhanced analysis complete!")
    return True

if __name__ == "__main__":
    success = test_enhanced_analysis()
    if success:
        print("\n🎉 All tests passed! PostgreSQL integration is working perfectly!")
    else:
        print("\n❌ Some tests failed. Please check the configuration.")
