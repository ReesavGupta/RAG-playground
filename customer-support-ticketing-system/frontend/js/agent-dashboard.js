class AgentDashboard {
    constructor() {
        this.ticketsList = document.getElementById('ticketsList');
        this.ticketDetailsSection = document.getElementById('ticketDetails');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.currentTicketId = null;
        
        this.initializeEventListeners();
        this.loadTickets();
    }
    
    initializeEventListeners() {
        document.getElementById('refreshBtn').addEventListener('click', () => this.loadTickets());
        document.getElementById('statusFilter').addEventListener('change', () => this.loadTickets());
        document.getElementById('categoryFilter').addEventListener('change', () => this.loadTickets());
        document.getElementById('priorityFilter').addEventListener('change', () => this.loadTickets());
        document.getElementById('resolveBtn').addEventListener('click', () => this.resolveTicket());
        document.getElementById('generateResponseBtn').addEventListener('click', () => this.generateResponse());
    }
    
    async loadTickets() {
        this.showLoading();
        
        try {
            const filters = this.getFilters();
            const queryString = new URLSearchParams(filters).toString();
            
            const response = await fetch(`/api/tickets/?${queryString}`);
            if (!response.ok) {
                throw new Error('Failed to load tickets');
            }
            
            const tickets = await response.json();
            this.displayTickets(tickets);
            
        } catch (error) {
            this.showError('Failed to load tickets');
            console.error('Error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    getFilters() {
        const filters = {};
        
        const status = document.getElementById('statusFilter').value;
        const category = document.getElementById('categoryFilter').value;
        const priority = document.getElementById('priorityFilter').value;
        
        if (status) filters.status = status;
        if (category) filters.category = category;
        if (priority) filters.priority = priority;
        
        return filters;
    }
    
    displayTickets(tickets) {
        if (tickets.length === 0) {
            this.ticketsList.innerHTML = '<p>No tickets found.</p>';
            return;
        }
        
        this.ticketsList.innerHTML = tickets.map(ticket => `
            <div class="ticket-card" onclick="agentDashboard.selectTicket(${ticket.id})">
                <div class="ticket-header">
                    <span class="ticket-number">#${ticket.ticket_number}</span>
                    <span class="ticket-priority priority-${ticket.priority}">${ticket.priority}</span>
                </div>
                <div class="ticket-subject">${ticket.subject}</div>
                <div class="ticket-meta">
                    <span class="ticket-status status-${ticket.status}">${ticket.status}</span>
                    <span>${ticket.category}</span>
                    <span>${new Date(ticket.created_at).toLocaleDateString()}</span>
                </div>
            </div>
        `).join('');
    }
    
    async selectTicket(ticketId) {
        this.currentTicketId = ticketId;
        this.showLoading();
        
        try {
            const response = await fetch(`/api/tickets/${ticketId}`);
            if (!response.ok) {
                throw new Error('Failed to load ticket details');
            }
            
            const ticket = await response.json();
            this.displayTicketDetails(ticket);
            
        } catch (error) {
            this.showError('Failed to load ticket details');
            console.error('Error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    displayTicketDetails(ticket) {
        const ticketInfo = document.getElementById('ticketInfo');
        
        ticketInfo.innerHTML = `
            <h3>Ticket #${ticket.ticket_number}</h3>
            <div class="ticket-detail">
                <p><strong>Customer:</strong> ${ticket.customer_name || 'N/A'} (${ticket.customer_email})</p>
                <p><strong>Subject:</strong> ${ticket.subject}</p>
                <p><strong>Description:</strong></p>
                <p>${ticket.description}</p>
                <p><strong>Category:</strong> ${ticket.category}</p>
                <p><strong>Priority:</strong> <span class="priority-${ticket.priority}">${ticket.priority}</span></p>
                <p><strong>Status:</strong> <span class="status-${ticket.status}">${ticket.status}</span></p>
                <p><strong>Created:</strong> ${new Date(ticket.created_at).toLocaleString()}</p>
                ${ticket.sentiment_score !== null ? `<p><strong>Sentiment:</strong> ${ticket.sentiment_score > 0 ? 'Positive' : ticket.sentiment_score < 0 ? 'Negative' : 'Neutral'}</p>` : ''}
            </div>
        `;
        
        // Show ticket details section
        this.ticketDetailsSection.classList.remove('hidden');
        
        // Clear any existing smart response
        document.getElementById('smartResponse').innerHTML = '';
        
        // Clear resolution text if ticket is already resolved
        if (ticket.status === 'resolved') {
            document.getElementById('resolutionText').value = ticket.resolution || '';
            document.getElementById('resolveBtn').disabled = true;
        } else {
            document.getElementById('resolutionText').value = '';
            document.getElementById('resolveBtn').disabled = false;
        }
    }
    
    async generateResponse() {
        if (!this.currentTicketId) return;
        
        this.showLoading();
        
        try {
            const response = await fetch(`/api/tickets/${this.currentTicketId}/response`, {
                method: 'POST'
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate response');
            }
            
            const responseData = await response.json();
            this.displaySmartResponse(responseData);
            
        } catch (error) {
            this.showError('Failed to generate smart response');
            console.error('Error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    displaySmartResponse(responseData) {
        const smartResponseDiv = document.getElementById('smartResponse');
        const confidence = responseData.confidence;
        const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';
        
        smartResponseDiv.innerHTML = `
            <h3>Smart Response Suggestion 
                <span class="confidence-score confidence-${confidenceClass}">
                    ${Math.round(confidence * 100)}% confidence
                </span>
            </h3>
            <div class="response-content">
                <p>${responseData.response}</p>
            </div>
            <div class="response-sources">
                <small>Based on ${responseData.sources.similar_tickets} similar tickets 
                and ${responseData.sources.knowledge_items} knowledge base items</small>
            </div>
            <button onclick="agentDashboard.useSmartResponse()" class="generate-btn">Use This Response</button>
        `;
        
        this.lastGeneratedResponse = responseData.response;
    }
    
    useSmartResponse() {
        if (this.lastGeneratedResponse) {
            document.getElementById('resolutionText').value = this.lastGeneratedResponse;
        }
    }
    
    async resolveTicket() {
        if (!this.currentTicketId) return;
        
        const resolution = document.getElementById('resolutionText').value.trim();
        if (!resolution) {
            this.showError('Please enter a resolution');
            return;
        }
        
        this.showLoading();
        
        try {
            const response = await fetch(`/api/tickets/${this.currentTicketId}/resolve`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    resolution: resolution,
                    agent_id: 'current_agent' // In real app, get from auth
                })
            });
            
            if (!response.ok) {
                throw new Error('Failed to resolve ticket');
            }
            
            this.showSuccessMessage('Ticket resolved successfully!');
            this.loadTickets(); // Refresh ticket list
            document.getElementById('resolveBtn').disabled = true;
            
        } catch (error) {
            this.showError('Failed to resolve ticket');
            console.error('Error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    showLoading() {
        this.loadingOverlay.classList.remove('hidden');
    }
    
    hideLoading() {
        this.loadingOverlay.classList.add('hidden');
    }
    
    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        
        document.querySelector('main').insertBefore(errorDiv, document.querySelector('main').firstChild);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
    
    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.textContent = message;
        
        document.querySelector('main').insertBefore(successDiv, document.querySelector('main').firstChild);
        
        setTimeout(() => {
            successDiv.remove();
        }, 5000);
    }
}

// Initialize when DOM is loaded
let agentDashboard;
document.addEventListener('DOMContentLoaded', () => {
    agentDashboard = new AgentDashboard();
});