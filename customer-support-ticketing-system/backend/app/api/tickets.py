from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from typing import List, Dict, Optional
from pydantic import BaseModel
from database.postgres_db import get_db
from services.tickets_service import TicketService
from services.rag_service import RAGService
from database.chroma_db import ChromaDBService
import os

router = APIRouter()

class TicketCreate(BaseModel):
    customer_email: str
    customer_name: Optional[str] = None
    subject: str
    description: str

class TicketResponse(BaseModel):
    ticket_id: int
    response: str
    confidence: float

# Initialize services
chroma_service = ChromaDBService()
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
rag_service = RAGService(openai_api_key=openai_api_key)

def get_ticket_service(db: Session = Depends(get_db)):
    return TicketService(db, chroma_service, rag_service)

@router.post("/", response_model=Dict)
async def create_ticket(
    ticket_data: TicketCreate,
    ticket_service: TicketService = Depends(get_ticket_service)
):
    """Create new support ticket"""
    try:
        result = ticket_service.create_ticket(ticket_data.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{ticket_id}", response_model=Dict)
async def get_ticket(
    ticket_id: int,
    ticket_service: TicketService = Depends(get_ticket_service)
):
    """Get ticket by ID"""
    try:
        ticket = ticket_service.get_ticket(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        return ticket
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[Dict])
async def list_tickets(
    status: Optional[str] = None,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
    ticket_service: TicketService = Depends(get_ticket_service)
):
    """List tickets with optional filters"""
    try:
        tickets = ticket_service.list_tickets(
            status=status,
            category=category,
            priority=priority,
            limit=limit,
            offset=offset
        )
        return tickets
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{ticket_id}/resolve")
async def resolve_ticket(
    ticket_id: int,
    resolution: str,
    agent_id: Optional[str] = None,
    ticket_service: TicketService = Depends(get_ticket_service)
):
    """Resolve ticket with solution"""
    try:
        result = ticket_service.update_ticket_resolution(
            ticket_id=ticket_id,
            resolution=resolution,
            agent_id=agent_id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{ticket_id}/response")
async def generate_response(
    ticket_id: int,
    ticket_service: TicketService = Depends(get_ticket_service)
):
    """Generate smart response for ticket"""
    try:
        ticket_instance = ticket_service.get_ticket_instance(ticket_id)
        if not ticket_instance:
            raise HTTPException(status_code=404, detail="Ticket not found")
        response_data = ticket_service.generate_smart_response(ticket_instance)
        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))