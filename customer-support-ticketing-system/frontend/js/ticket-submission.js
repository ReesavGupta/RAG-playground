class TicketSubmission {
    constructor() {
        this.form = document.getElementById('ticketForm');
        this.responseSection = document.getElementById('responseSection');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.form.addEventListener('submit', (e) => this.handleSubmit(e));
    }
    
    async handleSubmit(event) {
        event.preventDefault();
        
        const formData = new FormData(this.form);
        const ticketData = {
            customer_email: formData.get('customer_email'),
            customer_name: formData.get('customer_name'),
            subject: formData.get('subject'),
            description: formData.get('description')
        };
        
        this.showLoading();
        
        try {
            const response = await fetch('/api/tickets/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(ticketData)
            });
            
            if (!response.ok) {
                throw new Error('Failed to submit ticket');
            }
            
            const result = await response.json();
            this.displayTicketResult(result);
            
        } catch (error) {
            this.showError('Failed to submit ticket. Please try again.');
            console.error('Error:', error);
        } finally {
            this.hideLoading();
        }
    }
    
    displayTicketResult(result) {
        const ticketDetailsDiv = document.getElementById('ticketDetails');
        const suggestedResponseDiv = document.getElementById('suggestedResponse');
        
        // Display ticket details
        ticketDetailsDiv.innerHTML = `
            <h3>Ticket #${result.ticket.ticket_number}</h3>
            <div class="ticket-info">
                <p><strong>Subject:</strong> ${result.ticket.subject}</p>
                <p><strong>Category:</strong> ${result.ticket.category}</p>
                <p><strong>Priority:</strong> <span class="priority-${result.ticket.priority}">${result.ticket.priority}</span></p>
                <p><strong>Status:</strong> <span class="status-${result.ticket.status}">${result.ticket.status}</span></p>
                <p><strong>Created:</strong> ${new Date(result.ticket.created_at).toLocaleString()}</p>
            </div>
        `;
        
        // Display suggested response
        if (result.suggested_response) {
            const confidence = result.suggested_response.confidence;
            const confidenceClass = confidence > 0.7 ? 'high' : confidence > 0.4 ? 'medium' : 'low';
            
            suggestedResponseDiv.innerHTML = `
                <h3>Suggested Response 
                    <span class="confidence-score confidence-${confidenceClass}">
                        ${Math.round(confidence * 100)}% confidence
                    </span>
                </h3>
                <div class="response-content">
                    <p>${result.suggested_response.response}</p>
                </div>
                <div class="response-sources">
                    <small>Based on ${result.suggested_response.sources.similar_tickets} similar tickets 
                    and ${result.suggested_response.sources.knowledge_items} knowledge base items</small>
                </div>
            `;
        }
        
        // Show success message and hide form
        this.form.style.display = 'none';
        this.responseSection.classList.remove('hidden');
        
        // Show success message
        this.showSuccessMessage('Ticket submitted successfully!');
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
        
        this.form.insertBefore(errorDiv, this.form.firstChild);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }
    
    showSuccessMessage(message) {
        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.textContent = message;
        
        this.responseSection.insertBefore(successDiv, this.responseSection.firstChild);
        
        setTimeout(() => {
            successDiv.remove();
        }, 5000);
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TicketSubmission();
});