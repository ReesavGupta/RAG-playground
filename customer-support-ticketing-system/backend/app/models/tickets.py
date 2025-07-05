from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Boolean, ARRAY
from sqlalchemy.sql import func
from datetime import datetime
from app.database.postgres_db import Base

class Ticket(Base):
    __tablename__ = "tickets"
    
    id = Column(Integer, primary_key=True, index=True)
    ticket_number = Column(String(50), unique=True, index=True)
    customer_email = Column(String(255), index=True)
    customer_name = Column(String(255))
    subject = Column(String(500))
    description = Column(Text)
    category = Column(String(100), index=True)
    priority = Column(String(20), default="medium")
    status = Column(String(50), default="open", index=True)
    sentiment_score = Column(Float)
    confidence_score = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    resolved_at = Column(DateTime(timezone=True))
    assigned_agent = Column(String(255))
    resolution = Column(Text)
    customer_satisfaction = Column(Integer)

class KnowledgeBase(Base):
    __tablename__ = "knowledge_base"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(500))
    content = Column(Text)
    category = Column(String(100), index=True)
    tags = Column(ARRAY(String))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

class TicketCategory(Base):
    __tablename__ = "ticket_categories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, index=True)
    description = Column(Text)
    keywords = Column(ARRAY(String))
    priority_weight = Column(Integer, default=1)
    auto_response_template = Column(Text)