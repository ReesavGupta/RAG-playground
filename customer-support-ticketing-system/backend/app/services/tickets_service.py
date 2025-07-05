from typing import Dict, List, Optional
from sqlalchemy.orm import Session
from models.tickets import Ticket, KnowledgeBase
from services.categorization_service import TicketCategorizationService
from services.rag_service import RAGService
from database.chroma_db import ChromaDBService
from utils.text_processing import TextProcessor
import uuid
from datetime import datetime

class TicketService:
    def __init__(self, db_session: Session, chroma_service: ChromaDBService, 
                 rag_service: RAGService):
        self.db = db_session
        self.chroma = chroma_service
        self.rag = rag_service
        self.categorization = TicketCategorizationService()
        self.text_processor = TextProcessor()
    
    def create_ticket(self, ticket_data: Dict) -> Dict:
        """Create new support ticket with automatic categorization and basic escalation logic"""
        
        # Generate ticket number
        ticket_number = f"SUP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        
        # Categorize ticket
        category, cat_confidence = self.categorization.categorize_ticket(
            ticket_data['subject'], ticket_data['description']
        )
        
        # Determine priority
        priority = self.categorization.determine_priority(
            ticket_data['subject'], ticket_data['description']
        )
        
        # Extract sentiment
        sentiment = self.categorization.extract_sentiment(
            f"{ticket_data['subject']} {ticket_data['description']}"
        )
        
        # Basic escalation logic
        escalate = False
        if cat_confidence < 0.5 or sentiment < -0.5:
            escalate = True
        
        # Create ticket in database
        ticket = Ticket(
            ticket_number=ticket_number,
            customer_email=ticket_data['customer_email'],
            customer_name=ticket_data.get('customer_name', ''),
            subject=ticket_data['subject'],
            description=ticket_data['description'],
            category=category,
            priority=priority,
            sentiment_score=sentiment,
            confidence_score=cat_confidence,
            status='escalated' if escalate else 'open'
        )
        
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)
        
        # Add to vector database
        ticket_content = f"{ticket.subject} {ticket.description}"
        self.chroma.add_ticket(
            ticket_id=str(ticket.id),
            content=ticket_content,
            metadata={
                "ticket_number": ticket_number,
                "category": category,
                "priority": priority,
                "created_at": ticket.created_at.isoformat(),
                "escalated": escalate
            }
        )
        
        # Generate initial response
        response_data = self.generate_smart_response(ticket)
        
        return {
            "ticket": self._ticket_to_dict(ticket),
            "suggested_response": response_data,
            "escalated": escalate
        }
    
    def generate_smart_response(self, ticket: Ticket) -> Dict:
        """Generate smart response using RAG pipeline"""
        
        # Search for similar tickets
        query = f"{ticket.subject} {ticket.description}"
        similar_tickets = self.chroma.search_similar_tickets(query, n_results=3)
        
        # Search knowledge base
        kb_results = self.chroma.search_knowledge_base(query, n_results=2)
        
        # Fetch full ticket data for similar tickets
        similar_ticket_data = []
        if similar_tickets['ids']:
            for ticket_id in similar_tickets['ids'][0]:
                db_ticket = self.db.query(Ticket).filter(Ticket.id == int(ticket_id)).first()
                if db_ticket and db_ticket.resolution:  # type: ignore[reportGeneralTypeIssues]
                    similar_ticket_data.append({
                        "subject": db_ticket.subject,
                        "description": db_ticket.description,
                        "resolution": db_ticket.resolution,
                        "category": db_ticket.category
                    })
        
        # Fetch knowledge base items
        kb_items = []
        if kb_results['ids']:
            for kb_id in kb_results['ids'][0]:
                kb_item = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == int(kb_id)).first()
                if kb_item:
                    kb_items.append({
                        "title": kb_item.title,
                        "content": kb_item.content,
                        "category": kb_item.category
                    })
        
        # Generate response using RAG
        response_data = self.rag.generate_response(
            ticket_content=query,
            similar_tickets=similar_ticket_data,
            knowledge_items=kb_items
        )
        
        return response_data
    
    def update_ticket_resolution(self, ticket_id: int, resolution: str, agent_id: Optional[str] = None) -> Dict:
        """Update ticket with resolution"""
        ticket = self.db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket is None:
            raise ValueError("Ticket not found")

        ticket.resolution = resolution  # type: ignore
        ticket.status = "resolved"  # type: ignore
        ticket.resolved_at = datetime.now()  # type: ignore
        if agent_id is not None:
            ticket.assigned_agent = agent_id  # type: ignore

        self.db.commit()

        return self._ticket_to_dict(ticket)
    
    def _ticket_to_dict(self, ticket: Ticket) -> Dict:
        """Convert ticket object to dictionary"""
        return {
            "id": ticket.id,
            "ticket_number": ticket.ticket_number,
            "customer_email": ticket.customer_email,
            "customer_name": ticket.customer_name,
            "subject": ticket.subject,
            "description": ticket.description,
            "category": ticket.category,
            "priority": ticket.priority,
            "status": ticket.status,
            "sentiment_score": ticket.sentiment_score,
            "confidence_score": ticket.confidence_score,
            "created_at": ticket.created_at.isoformat() if ticket.created_at is not None else None,
            "resolved_at": ticket.resolved_at.isoformat() if ticket.resolved_at is not None else None,
            "assigned_agent": ticket.assigned_agent,
            "resolution": ticket.resolution
        }

    def get_ticket(self, ticket_id: int) -> Optional[Dict]:
        """Retrieve a ticket by its ID"""
        ticket = self.db.query(Ticket).filter(Ticket.id == ticket_id).first()
        if ticket is not None:  # type: ignore[reportGeneralTypeIssues]
            return self._ticket_to_dict(ticket)
        return None

    def get_ticket_instance(self, ticket_id: int) -> Optional[Ticket]:
        """Retrieve the Ticket ORM instance by its ID"""
        return self.db.query(Ticket).filter(Ticket.id == ticket_id).first()

    def list_tickets(self, status: Optional[str] = None, category: Optional[str] = None, priority: Optional[str] = None, limit: int = 20, offset: int = 0) -> List[Dict]:
        """List tickets with optional filters"""
        query = self.db.query(Ticket)
        if status:
            query = query.filter(Ticket.status == status)
        if category:
            query = query.filter(Ticket.category == category)
        if priority:
            query = query.filter(Ticket.priority == priority)
        tickets = query.order_by(Ticket.created_at.desc()).offset(offset).limit(limit).all()
        return [self._ticket_to_dict(ticket) for ticket in tickets]