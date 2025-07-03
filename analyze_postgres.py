#!/usr/bin/env python3
"""
Production ready script for using PostgreSQL with Business Intelligence Agent
Database: db_bisee
"""

from app import BusinessIntelligenceAgent
import json

def main():
    """Main function to analyze your PostgreSQL database"""
    print("🚀 Business Intelligence Analysis")
    print("Database: db_bisee (PostgreSQL)")
    print("=" * 50)
    
    # Your PostgreSQL configuration
    db_config = {
        "db_type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "db_bisee",
        "user": "postgres",
        "password": "password"
    }
    
    try:
        # Initialize the BI Agent
        agent = BusinessIntelligenceAgent(**db_config)
        
        if not agent.initialize():
            print("❌ Failed to connect to database")
            return
        
        # Get available tables
        tables = agent.get_available_tables()
        print(f"📊 Available tables: {tables}")
        
        # Analyze each table
        for table_name in tables:
            print(f"\n🔍 Analyzing table: {table_name}")
            print("-" * 40)
            
            # Generate comprehensive business report
            report = agent.generate_business_report(table_name)
            
            if 'executive_summary' in report:
                # Display Executive Summary
                summary = report['executive_summary']
                print("📋 EXECUTIVE SUMMARY:")
                print(f"  • Total Records: {summary['total_records']}")
                print(f"  • Data Dimensions: {summary['data_dimensions']}")
                print(f"  • Data Quality Score: {summary['data_quality_score']}%")
                
                # Display Key Findings
                print("\n💡 KEY FINDINGS:")
                for i, finding in enumerate(summary['key_findings'], 1):
                    print(f"  {i}. {finding}")
                
                # Display Recommendations
                print("\n🎯 RECOMMENDATIONS:")
                for rec in report['recommendations']:
                    print(f"  • {rec['action']} ({rec['priority']} priority)")
                    print(f"    Details: {rec['details']}")
                    print(f"    Timeline: {rec['timeline']}")
                
                # Sample Data Preview
                sample_data = agent.db_connector.get_sample_data(table_name, 5)
                print(f"\n📊 SAMPLE DATA (first 5 rows):")
                print(sample_data.to_string(index=False))
                
                print("\n" + "=" * 60)
        
        # Close connection
        agent.db_connector.close()
        print("\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please check your database connection.")

def analyze_specific_table(table_name: str):
    """Analyze a specific table in detail"""
    db_config = {
        "db_type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "db_bisee",
        "user": "postgres",
        "password": "password"
    }
    
    try:
        agent = BusinessIntelligenceAgent(**db_config)
        
        if agent.db_connector.connect():
            # Get detailed analysis
            analysis = agent.analyze_table(table_name)
            
            # Save to JSON file
            output_file = f"{table_name}_analysis.json"
            with open(output_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"✅ Detailed analysis saved to {output_file}")
            
            agent.db_connector.close()
        
    except Exception as e:
        print(f"❌ Error analyzing table {table_name}: {e}")

if __name__ == "__main__":
    # Run main analysis
    main()
    
    # Uncomment the line below to analyze a specific table and save to JSON
    # analyze_specific_table("TransactionData")
