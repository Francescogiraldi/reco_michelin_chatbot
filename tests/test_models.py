"""Tests for data models."""

import pytest
from decimal import Decimal
from pydantic import ValidationError

from src.models.tire import TireProduct, TireCatalog, TireCategory, ChatMessage, ChatSession


class TestTireProduct:
    """Test TireProduct model."""
    
    def test_valid_tire_product(self):
        """Test creating a valid tire product."""
        product = TireProduct(
            id="TEST-001",
            name="Test Tire",
            description="A test tire for unit testing purposes",
            category=TireCategory.SUMMER,
            price=Decimal("150.00"),
            link="https://example.com/tire"
        )
        
        assert product.id == "TEST-001"
        assert product.name == "Test Tire"
        assert product.category == TireCategory.SUMMER
        assert product.price == Decimal("150.00")
    
    def test_invalid_price(self):
        """Test validation of invalid price."""
        with pytest.raises(ValidationError):
            TireProduct(
                id="TEST-001",
                name="Test Tire",
                description="A test tire",
                category=TireCategory.SUMMER,
                price=Decimal("-10.00"),  # Invalid negative price
                link="https://example.com/tire"
            )
    
    def test_invalid_url(self):
        """Test validation of invalid URL."""
        with pytest.raises(ValidationError):
            TireProduct(
                id="TEST-001",
                name="Test Tire",
                description="A test tire",
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="not-a-valid-url"  # Invalid URL
            )
    
    def test_empty_description(self):
        """Test validation of empty description."""
        with pytest.raises(ValidationError):
            TireProduct(
                id="TEST-001",
                name="Test Tire",
                description="",  # Empty description
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="https://example.com/tire"
            )
    
    def test_to_search_content(self):
        """Test search content generation."""
        product = TireProduct(
            id="TEST-001",
            name="Test Tire",
            description="A high-performance summer tire",
            category=TireCategory.SUMMER,
            price=Decimal("150.00"),
            link="https://example.com/tire",
            width=225,
            aspect_ratio=45,
            rim_diameter=17
        )
        
        content = product.to_search_content()
        assert "Test Tire" in content
        assert "A high-performance summer tire" in content
        assert "Catégorie: Été" in content
        assert "Prix: 150.0€" in content
        assert "Dimension: 225/45R17" in content
    
    def test_optional_fields(self):
        """Test optional technical fields."""
        product = TireProduct(
            id="TEST-001",
            name="Test Tire",
            description="A test tire with technical specs",
            category=TireCategory.SUMMER,
            price=Decimal("150.00"),
            link="https://example.com/tire",
            width=225,
            aspect_ratio=45,
            rim_diameter=17,
            wet_grip="A",
            fuel_efficiency="B",
            noise_level=70
        )
        
        assert product.width == 225
        assert product.wet_grip == "A"
        assert product.fuel_efficiency == "B"
        assert product.noise_level == 70


class TestTireCatalog:
    """Test TireCatalog model."""
    
    def test_valid_catalog(self):
        """Test creating a valid catalog."""
        products = [
            TireProduct(
                id="TEST-001",
                name="Test Tire 1",
                description="First test tire",
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="https://example.com/tire1"
            ),
            TireProduct(
                id="TEST-002",
                name="Test Tire 2",
                description="Second test tire",
                category=TireCategory.WINTER,
                price=Decimal("180.00"),
                link="https://example.com/tire2"
            )
        ]
        
        catalog = TireCatalog(products=products, total_count=2)
        assert len(catalog.products) == 2
        assert catalog.total_count == 2
    
    def test_invalid_total_count(self):
        """Test validation of mismatched total count."""
        products = [
            TireProduct(
                id="TEST-001",
                name="Test Tire",
                description="A test tire",
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="https://example.com/tire"
            )
        ]
        
        with pytest.raises(ValidationError):
            TireCatalog(products=products, total_count=5)  # Wrong count
    
    def test_get_by_id(self):
        """Test getting product by ID."""
        products = [
            TireProduct(
                id="TEST-001",
                name="Test Tire 1",
                description="First test tire",
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="https://example.com/tire1"
            )
        ]
        
        catalog = TireCatalog(products=products, total_count=1)
        product = catalog.get_by_id("TEST-001")
        assert product is not None
        assert product.name == "Test Tire 1"
        
        # Test non-existent ID
        assert catalog.get_by_id("NON-EXISTENT") is None
    
    def test_get_by_category(self):
        """Test getting products by category."""
        products = [
            TireProduct(
                id="TEST-001",
                name="Summer Tire",
                description="A summer tire",
                category=TireCategory.SUMMER,
                price=Decimal("150.00"),
                link="https://example.com/tire1"
            ),
            TireProduct(
                id="TEST-002",
                name="Winter Tire",
                description="A winter tire",
                category=TireCategory.WINTER,
                price=Decimal("180.00"),
                link="https://example.com/tire2"
            )
        ]
        
        catalog = TireCatalog(products=products, total_count=2)
        summer_tires = catalog.get_by_category(TireCategory.SUMMER)
        assert len(summer_tires) == 1
        assert summer_tires[0].name == "Summer Tire"
    
    def test_get_in_price_range(self):
        """Test getting products in price range."""
        products = [
            TireProduct(
                id="TEST-001",
                name="Cheap Tire",
                description="An affordable tire",
                category=TireCategory.SUMMER,
                price=Decimal("100.00"),
                link="https://example.com/tire1"
            ),
            TireProduct(
                id="TEST-002",
                name="Expensive Tire",
                description="A premium tire",
                category=TireCategory.SUMMER,
                price=Decimal("300.00"),
                link="https://example.com/tire2"
            )
        ]
        
        catalog = TireCatalog(products=products, total_count=2)
        affordable_tires = catalog.get_in_price_range(Decimal("50.00"), Decimal("150.00"))
        assert len(affordable_tires) == 1
        assert affordable_tires[0].name == "Cheap Tire"


class TestChatMessage:
    """Test ChatMessage model."""
    
    def test_valid_message(self):
        """Test creating a valid chat message."""
        message = ChatMessage(
            role="user",
            content="Hello, I need help with tires"
        )
        
        assert message.role == "user"
        assert message.content == "Hello, I need help with tires"
    
    def test_invalid_role(self):
        """Test validation of invalid role."""
        with pytest.raises(ValidationError):
            ChatMessage(
                role="invalid_role",  # Invalid role
                content="Hello"
            )
    
    def test_empty_content(self):
        """Test validation of empty content."""
        with pytest.raises(ValidationError):
            ChatMessage(
                role="user",
                content=""  # Empty content
            )


class TestChatSession:
    """Test ChatSession model."""
    
    def test_valid_session(self):
        """Test creating a valid chat session."""
        session = ChatSession(session_id="test-session-123")
        assert session.session_id == "test-session-123"
        assert len(session.messages) == 0
    
    def test_add_message(self):
        """Test adding messages to session."""
        session = ChatSession(session_id="test-session-123")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Hello"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "Hi there!"
    
    def test_get_recent_messages(self):
        """Test getting recent messages."""
        session = ChatSession(session_id="test-session-123")
        
        # Add multiple messages
        for i in range(10):
            session.add_message("user", f"Message {i}")
        
        # Get recent messages
        recent = session.get_recent_messages(3)
        assert len(recent) == 3
        assert recent[0].content == "Message 7"
        assert recent[2].content == "Message 9"
    
    def test_clear_messages(self):
        """Test clearing messages."""
        session = ChatSession(session_id="test-session-123")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi!")
        
        assert len(session.messages) == 2
        
        session.clear_messages()
        assert len(session.messages) == 0