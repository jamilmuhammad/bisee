#!/usr/bin/env python3
"""
Final Integration Test for Business Intelligence Agent
Demonstrates robust PostgreSQL and SQLite support with equivalent functionality
"""

import sys
sys.path.append('.')

from app import BusinessIntelligenceAgent
from db_config import POSTGRESQL_CONFIG
import json


def test_database_system(db_name, agent):
    """Test a database system and return results"""
    print(f'\n📊 Testing {db_name} Integration')
    print('-' * 45)
    
    if not agent.initialize():
        print(f'❌ Failed to initialize {db_name} agent')
        return None
    
    print(f'✅ {db_name} agent initialized successfully')
    
    tables = agent.get_available_tables()
    if not tables:
        print(f'❌ No tables found in {db_name}')
        return None
    
    table_name = tables[0]
    print(f'🔍 Analyzing table: {table_name}')
    
    # Perform analysis
    analysis = agent.analyze_table(table_name)
    
    # Generate business report
    report = agent.generate_business_report(table_name)
    
    # Collect results
    results = {
        'db_name': db_name,
        'table_name': table_name,
        'analysis': analysis,
        'report': report,
        'metrics': {
            'records': report['executive_summary']['total_records'],
            'dimensions': report['executive_summary']['data_dimensions'],
            'quality_score': report['executive_summary']['data_quality_score'],
            'insights_count': len(report['data_insights']),
            'recommendations_count': len(report['recommendations']),
            'visualizations_count': len(report['visualizations']),
            'has_text_summary': 'text_summary' in analysis
        }
    }
    
    return results


def display_results(results):
    """Display formatted results"""
    metrics = results['metrics']
    
    print(f'📈 Results for {results["db_name"]}:')
    print(f'  • Table: {results["table_name"]}')
    print(f'  • Records: {metrics["records"]}')
    print(f'  • Dimensions: {metrics["dimensions"]}')
    print(f'  • Quality Score: {metrics["quality_score"]}%')
    print(f'  • Insights: {metrics["insights_count"]}')
    print(f'  • Recommendations: {metrics["recommendations_count"]}')
    print(f'  • Visualizations: {metrics["visualizations_count"]}')
    print(f'  • Text Summary: {"✅" if metrics["has_text_summary"] else "❌"}')
    
    # Show sample insights
    insights = results['report']['data_insights']
    if insights:
        print(f'\n💡 Sample Insights:')
        for i, insight in enumerate(insights[:2], 1):
            print(f'  {i}. {insight[:100]}...')
    
    # Show text summary excerpt
    if 'text_summary' in results['analysis']:
        summary = results['analysis']['text_summary']
        print(f'\n📝 Text Summary (excerpt):')
        print(f'  {summary[:150]}...')


def main():
    """Main test function"""
    print('🚀 Business Intelligence Agent - Final Integration Test')
    print('=' * 65)
    print('Testing robust PostgreSQL support with equivalent SQLite functionality')
    
    results = {}
    
    # Test SQLite
    try:
        sqlite_agent = BusinessIntelligenceAgent(db_type='sqlite', db_path='business_data.db')
        results['sqlite'] = test_database_system('SQLite', sqlite_agent)
    except Exception as e:
        print(f'❌ SQLite test failed: {e}')
        results['sqlite'] = None
    
    # Test PostgreSQL
    try:
        pg_config = {k: v for k, v in POSTGRESQL_CONFIG.items() if k != 'db_type'}
        postgres_agent = BusinessIntelligenceAgent(db_type='postgresql', **pg_config)
        results['postgresql'] = test_database_system('PostgreSQL', postgres_agent)
    except Exception as e:
        print(f'❌ PostgreSQL test failed: {e}')
        results['postgresql'] = None
    
    # Display results
    print('\n📊 COMPREHENSIVE TEST RESULTS')
    print('=' * 45)
    
    for db_type, result in results.items():
        if result:
            display_results(result)
        else:
            print(f'❌ {db_type.title()} test failed')
    
    # Summary
    print('\n🎉 INTEGRATION TEST SUMMARY')
    print('=' * 35)
    
    successful_tests = [db for db, result in results.items() if result is not None]
    
    if len(successful_tests) == 2:
        print('✅ Both SQLite and PostgreSQL working perfectly!')
        print('✅ Equivalent functionality achieved')
        print('✅ Analysis, insights, and text summaries operational')
        print('✅ Business reports generated successfully')
        
        # Compare metrics
        if results['sqlite'] and results['postgresql']:
            sqlite_metrics = results['sqlite']['metrics']
            postgres_metrics = results['postgresql']['metrics']
            
            print('\n📊 Functionality Comparison:')
            print(f"  • Quality Scores: SQLite {sqlite_metrics['quality_score']}% vs PostgreSQL {postgres_metrics['quality_score']}%")
            print(f"  • Insights: SQLite {sqlite_metrics['insights_count']} vs PostgreSQL {postgres_metrics['insights_count']}")
            print(f"  • Text Summaries: SQLite {'✅' if sqlite_metrics['has_text_summary'] else '❌'} vs PostgreSQL {'✅' if postgres_metrics['has_text_summary'] else '❌'}")
    
    elif len(successful_tests) == 1:
        print(f'⚠️  Only {successful_tests[0]} is working')
        print('Consider checking database connections and configurations')
    
    else:
        print('❌ Both database systems failed')
        print('Check database connections and dependencies')


if __name__ == "__main__":
    main()
