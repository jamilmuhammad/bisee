# Business Intelligence AI Agent System

A comprehensive business intelligence system that supports both SQLite and PostgreSQL databases for data analysis, visualization, and automated insights generation.

## Features

- **Multi-Database Support**: Works with both SQLite and PostgreSQL databases
- **Automated Data Analysis**: Comprehensive table analysis with data quality metrics
- **Business Insights Generation**: AI-powered insight generation with actionable recommendations
- **SQL Query Generation**: Automatic generation of optimized SQL queries for analysis
- **Visualization Support**: Matplotlib-based visualization generation
- **Schema Analysis**: Detailed database schema inspection and documentation

## Installation

1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Database Support

### SQLite (Default)
SQLite is the default database type and requires no additional setup:

```python
from app import BusinessIntelligenceAgent

# SQLite configuration
agent = BusinessIntelligenceAgent(
    db_type="sqlite",
    db_path="business_data.db"
)
```

### PostgreSQL
PostgreSQL support requires a running PostgreSQL instance:

```python
from app import BusinessIntelligenceAgent

# PostgreSQL configuration
agent = BusinessIntelligenceAgent(
    db_type="postgresql",
    host="localhost",
    port=5432,
    database="business_data",
    user="postgres",
    password="your_password"
)
```

## Quick Start

### Using SQLite (Default)
```python
from app import BusinessIntelligenceAgent

# Initialize with SQLite
agent = BusinessIntelligenceAgent()

# Connect and create sample data
if agent.initialize():
    # Get available tables
    tables = agent.get_available_tables()
    print(f"Available tables: {tables}")
    
    # Analyze a table
    if tables:
        report = agent.generate_business_report(tables[0])
        print(report['executive_summary'])
```

### Using PostgreSQL
```python
from app import BusinessIntelligenceAgent

# Initialize with PostgreSQL
agent = BusinessIntelligenceAgent(
    db_type="postgresql",
    host="localhost",
    database="business_data",
    user="postgres",
    password="your_password"
)

# Connect and analyze
if agent.initialize():
    tables = agent.get_available_tables()
    if tables:
        report = agent.generate_business_report(tables[0])
        print(report['executive_summary'])
```

## Configuration Examples

### Environment Variables
Set database configuration using environment variables:

```bash
export DB_TYPE="postgresql"
export DB_HOST="localhost"
export DB_PORT="5432"
export DB_NAME="business_data"
export DB_USER="postgres"
export DB_PASSWORD="your_password"
```

Then use:
```python
from db_config import get_db_config_from_env
from app import BusinessIntelligenceAgent

config = get_db_config_from_env()
agent = BusinessIntelligenceAgent(**config)
```

### Configuration Files
Use predefined configurations from `db_config.py`:

```python
from db_config import POSTGRESQL_CONFIG
from app import BusinessIntelligenceAgent

agent = BusinessIntelligenceAgent(**POSTGRESQL_CONFIG)
```

## PostgreSQL Setup

### Local Installation
1. Install PostgreSQL:
   ```bash
   # macOS with Homebrew
   brew install postgresql
   brew services start postgresql
   
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib
   sudo systemctl start postgresql
   ```

2. Create database and user:
   ```sql
   CREATE DATABASE business_data;
   CREATE USER business_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE business_data TO business_user;
   ```

### Cloud PostgreSQL
The system works with cloud PostgreSQL services:

- **AWS RDS**: Use the RDS endpoint as host
- **Google Cloud SQL**: Use the Cloud SQL instance IP
- **Azure Database**: Use the Azure PostgreSQL server name
- **Heroku Postgres**: Use the Heroku database URL (parse connection string)

Example for AWS RDS:
```python
agent = BusinessIntelligenceAgent(
    db_type="postgresql",
    host="myinstance.region.rds.amazonaws.com",
    port=5432,
    database="business_data",
    user="dbuser",
    password="dbpassword"
)
```

## Database Schema Differences

The system automatically handles differences between SQLite and PostgreSQL:

### SQLite
- Uses `INTEGER PRIMARY KEY` for auto-increment
- Uses `TEXT` and `REAL` data types
- Uses `PRAGMA` commands for schema inspection

### PostgreSQL
- Uses `SERIAL PRIMARY KEY` for auto-increment
- Uses `VARCHAR`, `DECIMAL`, `TIMESTAMP` data types
- Uses `information_schema` for schema inspection
- Supports `ON CONFLICT` for upsert operations

## Key Classes

### DatabaseConnector
Handles database connections and low-level operations:
- Multi-database connection management
- Schema inspection
- Query execution
- Context manager support (`with` statement)

### DataAnalyzer
Performs comprehensive data analysis:
- Data quality assessment
- Statistical summaries
- Pattern identification
- Business insight generation

### QueryGenerator
Generates optimized SQL queries:
- Summary queries
- Categorical analysis
- Numeric analysis
- Time series analysis

### VisualizationGenerator
Creates visualization configurations:
- Chart type determination
- Data formatting
- Styling configuration
- Matplotlib integration

### BusinessIntelligenceAgent
Main orchestrator class:
- Coordinates all components
- Manages initialization
- Generates comprehensive reports

## Sample Data

The system automatically creates sample transaction data for demonstration:

| Field | SQLite Type | PostgreSQL Type | Description |
|-------|-------------|-----------------|-------------|
| TransactionID | INTEGER PRIMARY KEY | SERIAL PRIMARY KEY | Unique identifier |
| Transaction Type | TEXT | VARCHAR(50) | Type of transaction |
| Amount | REAL | DECIMAL(10,2) | Transaction amount |
| Transaction Status | TEXT | VARCHAR(20) | Success/Failed status |
| Fraud Flag | TEXT | VARCHAR(5) | True/False fraud indicator |
| Timestamp | TEXT | TIMESTAMP | Transaction timestamp |
| Device Used | TEXT | VARCHAR(20) | Device type |
| Geolocation | TEXT | VARCHAR(50) | Latitude,Longitude |
| Latency ms | INTEGER | INTEGER | Network latency |
| Slice Bandwidth Mbps | REAL | DECIMAL(5,1) | Bandwidth measurement |

## Running Examples

### Basic Example
```bash
python app.py
```

### PostgreSQL Example
```bash
python postgresql_example.py
```

### Custom Configuration
```python
from app import BusinessIntelligenceAgent

# Custom configuration
agent = BusinessIntelligenceAgent(
    db_type="postgresql",
    host="your-db-host.com",
    port=5432,
    database="your_database",
    user="your_username",
    password="your_password"
)

if agent.initialize():
    print("✅ Connected successfully!")
    tables = agent.get_available_tables()
    for table in tables:
        analysis = agent.analyze_table(table)
        print(f"Table {table}: {analysis['schema']['row_count']} rows")
```

## Error Handling

The system includes comprehensive error handling:

- Database connection failures
- Missing dependencies
- Invalid configurations
- Query execution errors
- Data type conversion issues

## Dependencies

- `pandas>=1.5.0`: Data manipulation and analysis
- `numpy>=1.24.0`: Numerical computing
- `matplotlib>=3.6.0`: Plotting and visualization
- `seaborn>=0.12.0`: Statistical data visualization
- `psycopg2-binary>=2.9.0`: PostgreSQL database adapter

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is available under the MIT License.

## Support

For issues and questions:
1. Check the error messages for specific guidance
2. Ensure database connectivity
3. Verify dependency installation
4. Review configuration parameters

Common issues:
- **Import errors**: Install missing dependencies with `pip install -r requirements.txt`
- **Connection failures**: Check database credentials and network connectivity
- **Permission errors**: Ensure database user has required privileges
