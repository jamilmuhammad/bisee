# Database Configuration Examples

# SQLite Configuration (default)
SQLITE_CONFIG = {
    "db_type": "sqlite",
    "db_path": "business_data.db"
}

# PostgreSQL Configuration
POSTGRESQL_CONFIG = {
    "db_type": "postgresql",
    "host": "localhost",
    "port": 5432,
    "database": "db_bisee",
    "user": "postgres",
    "password": "password"
}

# PostgreSQL Cloud Configuration (e.g., AWS RDS, Google Cloud SQL)
POSTGRESQL_CLOUD_CONFIG = {
    "db_type": "postgresql",
    "host": "your-db-instance.region.rds.amazonaws.com",
    "port": 5432,
    "database": "business_data",
    "user": "your_username",
    "password": "your_password"
}

# Environment-based configuration
import os

def get_db_config_from_env():
    """Get database configuration from environment variables"""
    db_type = os.getenv("DB_TYPE", "sqlite")
    
    if db_type == "postgresql":
        return {
            "db_type": "postgresql",
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "business_data"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
    else:
        return {
            "db_type": "sqlite",
            "db_path": os.getenv("DB_PATH", "business_data.db")
        }

# Usage examples:
# 
# # Using SQLite
# from app import BusinessIntelligenceAgent
# agent = BusinessIntelligenceAgent(**SQLITE_CONFIG)
# 
# # Using PostgreSQL
# agent = BusinessIntelligenceAgent(**POSTGRESQL_CONFIG)
# 
# # Using environment variables
# agent = BusinessIntelligenceAgent(**get_db_config_from_env())
