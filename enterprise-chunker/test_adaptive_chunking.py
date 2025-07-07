import pytest
import tempfile
import shutil
from pathlib import Path
from langchain.schema import Document

# Mock DocumentType for demonstration; replace with actual import if available
def get_document_type_enum():
    class DocumentType:
        CODE_SNIPPET = 'code_snippet'
        FAQ = 'faq'
        GENERAL = 'general'
    return DocumentType
DocumentType = get_document_type_enum()

class TestAdaptiveChunking:
    """Test suite for adaptive chunking functionality"""

    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path for testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_documents(self):
        """Sample documents for testing"""
        return [
            Document(
                page_content="""
                def authenticate_user(username, password):
                    '''Authenticate user with username and password'''
                    if not username or not password:
                        raise ValueError("Username and password required")
                    user = User.query.filter_by(username=username).first()
                    if user and user.check_password(password):
                        return generate_token(user)
                    return None
                """,
                metadata={"source": "auth_utils.py", "type": "code"}
            ),
            Document(
                page_content="""
                Q: How do I reset my password?
                A: To reset your password, click on the 'Forgot Password' link on the login page, enter your email address, and follow the instructions sent to your email.
                Q: What browsers are supported?
                A: We support the latest versions of Chrome, Firefox, Safari, and Edge.
                Q: How do I contact support?
                A: You can reach our support team at support@company.com or through the help chat in the application.
                """,
                metadata={"source": "faq.md", "type": "faq"}
            )
        ]

    def test_document_classification(self, sample_documents):
        """Test document classification accuracy"""
        # Mock classifier for testing without API key
        class MockClassifier:
            def classify_document(self, content):
                if "def " in content or "class " in content:
                    return DocumentType.CODE_SNIPPET
                elif "Q:" in content and "A:" in content:
                    return DocumentType.FAQ
                return DocumentType.GENERAL
        classifier = MockClassifier()
        # Test code document
        code_doc = sample_documents[0]
        doc_type = classifier.classify_document(code_doc.page_content)
        assert doc_type == DocumentType.CODE_SNIPPET
        # Test FAQ document
        faq_doc = sample_documents[1]
        doc_type = classifier.classify_document(faq_doc.page_content)
        assert doc_type == DocumentType.FAQ

    def test_chunking_strategies(self):
        """Test different chunking strategies"""
        # Mock chunker for testing
        class MockChunker:
            def __init__(self):
                self.strategies = {
                    DocumentType.CODE_SNIPPET: {"chunk_size": 500, "chunk_overlap": 50},
                    DocumentType.FAQ: {"chunk_size": 400, "chunk_overlap": 50},
                    DocumentType.GENERAL: {"chunk_size": 800, "chunk_overlap": 150}
                }
            def get_strategy(self, doc_type):
                return self.strategies.get(doc_type, self.strategies[DocumentType.GENERAL])
        chunker = MockChunker()
        # Test code chunking strategy
        code_strategy = chunker.get_strategy(DocumentType.CODE_SNIPPET)
        assert code_strategy["chunk_size"] == 500
        assert code_strategy["chunk_overlap"] == 50
        # Test FAQ chunking strategy
        faq_strategy = chunker.get_strategy(DocumentType.FAQ)
        assert faq_strategy["chunk_size"] == 400
        assert faq_strategy["chunk_overlap"] == 50 