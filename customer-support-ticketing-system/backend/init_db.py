#!/usr/bin/env python3
"""
Database initialization script for the Customer Support Ticketing System
"""
from app.database.postgres_db import engine, Base
from app.models.tickets import Ticket  # Import models to register them

def init_db():
    """Initialize the database by creating all tables"""
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully!")

if __name__ == "__main__":
    init_db() 