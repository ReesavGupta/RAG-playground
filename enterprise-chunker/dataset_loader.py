import pandas as pd
from pathlib import Path
from typing import List
from langchain.schema import Document

class DatasetLoader:
    """Load datasets from various sources for testing"""
    def __init__(self):
        self.cache_dir = Path("./dataset_cache")
        self.cache_dir.mkdir(exist_ok=True)
    def load_sample_enterprise_data(self) -> List[Document]:
        """Load sample enterprise data for testing"""
        tech_docs = [
            {
                'content': """
                # Database Connection Configuration
                ## Overview
                This document describes the database connection configuration for our enterprise application.
                ## Connection Parameters
                - **Host**: The database server hostname or IP address
                - **Port**: Database server port (default: 5432 for PostgreSQL)
                - **Database**: Target database name
                - **Username**: Database user account
                - **Password**: User account password
                ## Configuration Example
                ```yaml
                database:
                  host: db.company.com
                  port: 5432
                  database: enterprise_db
                  username: app_user
                  password: ${DB_PASSWORD}
                  ssl_mode: require
                ```
                ## Connection Pooling
                Configure connection pooling for optimal performance:
                - **pool_size**: Number of connections to maintain (default: 10)
                - **max_overflow**: Maximum overflow connections (default: 20)
                - **pool_timeout**: Connection timeout in seconds (default: 30)
                """,
                'metadata': {'source': 'db_config.md', 'document_type': 'technical_documentation'}
            },
            {
                'content': """
                # API Rate Limiting Guide
                ## Rate Limit Structure
                Our API implements rate limiting to ensure fair usage:
                ### Limits by Plan
                - **Free Plan**: 100 requests per hour
                - **Pro Plan**: 1,000 requests per hour  
                - **Enterprise Plan**: 10,000 requests per hour
                ### Rate Limit Headers
                Every API response includes rate limit information:
                ```
                X-RateLimit-Limit: 1000
                X-RateLimit-Remaining: 999
                X-RateLimit-Reset: 1609459200
                ```
                ### Handling Rate Limits
                When you exceed the rate limit, the API returns a 429 status code:
                ```json
                {
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 3600
                }
                ```
                ### Best Practices
                1. Implement exponential backoff for retries
                2. Cache responses when possible
                3. Use webhooks instead of polling
                4. Monitor your usage patterns
                """,
                'metadata': {'source': 'api_rate_limiting.md', 'document_type': 'api_reference'}
            }
        ]
        documents = []
        for doc_data in tech_docs:
            doc = Document(
                page_content=doc_data['content'],
                metadata=doc_data['metadata']
            )
            documents.append(doc)
        return documents
    def create_synthetic_support_tickets(self, num_tickets: int = 10) -> List[Document]:
        """Create synthetic support tickets for testing"""
        ticket_templates = [
            {
                'title': 'Login Issues',
                'content': """
                Ticket #{ticket_id}
                Status: Open
                Priority: High
                Customer: {customer}
                Issue: Users unable to login to the application
                Description:
                Multiple users are reporting login failures since the system update yesterday.
                Error message: "Invalid credentials" even with correct username/password.
                Steps to Reproduce:
                1. Navigate to login page
                2. Enter valid credentials
                3. Click login button
                4. Receive error message
                Impact: 50+ users affected
                Urgency: Critical - blocking user access
                """
            },
            {
                'title': 'Performance Degradation',
                'content': """
                Ticket #{ticket_id}
                Status: In Progress
                Priority: Medium
                Customer: {customer}
                Issue: Application response times significantly slower
                Description:
                Page load times have increased from 2-3 seconds to 15-20 seconds.
                Issue started around 2 PM EST today.
                Affected Areas:
                - Dashboard loading
                - Report generation
                - Search functionality
                Current Investigation:
                - Checking database query performance
                - Reviewing server resource usage
                - Analyzing recent code deployments
                """
            }
        ]
        documents = []
        for i in range(num_tickets):
            template = ticket_templates[i % len(ticket_templates)]
            content = template['content'].format(
                ticket_id=f'SUP-{1000 + i}',
                customer=f'Customer{i + 1}'
            )
            doc = Document(
                page_content=content,
                metadata={
                    'source': f'support_ticket_{1000 + i}.txt',
                    'document_type': 'support_ticket',
                    'ticket_id': f'SUP-{1000 + i}'
                }
            )
            documents.append(doc)
        return documents 