"""Tire product models for data validation."""

from typing import Optional, List
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field, HttpUrl, validator


class TireCategory(str, Enum):
    """Tire category enumeration."""
    SUMMER = "Été"
    WINTER = "Hiver"
    ALL_SEASON = "Toutes-saisons"
    TOURISM = "Tourisme"
    SUV_SUMMER = "SUV Été"
    SUV_ALL_SEASON = "SUV Toutes-saisons"
    ECO = "Éco"
    SPORT = "Sport"
    PICKUP_SUV = "Pickup/SUV"
    WINTER_NORDIC = "Hiver Nordique"


class TireProduct(BaseModel):
    """Tire product model with validation."""
    
    id: str = Field(..., min_length=1, max_length=50, description="Unique product identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Product name")
    description: str = Field(..., min_length=10, max_length=1000, description="Product description")
    category: TireCategory = Field(..., description="Tire category")
    price: Decimal = Field(..., gt=0, le=10000, description="Price in euros")
    link: HttpUrl = Field(..., description="Product URL")
    
    # Optional fields for enhanced functionality
    width: Optional[int] = Field(None, ge=125, le=355, description="Tire width in mm")
    aspect_ratio: Optional[int] = Field(None, ge=25, le=85, description="Aspect ratio")
    rim_diameter: Optional[int] = Field(None, ge=13, le=24, description="Rim diameter in inches")
    load_index: Optional[int] = Field(None, ge=60, le=150, description="Load index")
    speed_rating: Optional[str] = Field(None, pattern=r"^[A-Z]$", description="Speed rating")
    
    # Performance characteristics
    wet_grip: Optional[str] = Field(None, pattern=r"^[A-G]$", description="EU wet grip rating")
    fuel_efficiency: Optional[str] = Field(None, pattern=r"^[A-G]$", description="EU fuel efficiency rating")
    noise_level: Optional[int] = Field(None, ge=60, le=80, description="Noise level in dB")
    
    # Availability and stock
    in_stock: bool = Field(default=True, description="Product availability")
    stock_quantity: Optional[int] = Field(None, ge=0, description="Stock quantity")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "forbid"
    
    @validator('description')
    def validate_description(cls, v):
        """Validate description content."""
        if not v.strip():
            raise ValueError('Description cannot be empty')
        return v.strip()
    
    @validator('price')
    def validate_price(cls, v):
        """Validate price format."""
        if v <= 0:
            raise ValueError('Price must be positive')
        # Round to 2 decimal places
        return round(v, 2)
    
    def to_search_content(self) -> str:
        """Generate content for vector search indexing."""
        content_parts = [
            self.name,
            self.description,
            f"Catégorie: {self.category}",
            f"Prix: {self.price}€"
        ]
        
        # Add technical specifications if available
        if self.width and self.aspect_ratio and self.rim_diameter:
            content_parts.append(f"Dimension: {self.width}/{self.aspect_ratio}R{self.rim_diameter}")
        
        if self.wet_grip:
            content_parts.append(f"Adhérence sur sol mouillé: {self.wet_grip}")
        
        if self.fuel_efficiency:
            content_parts.append(f"Efficacité énergétique: {self.fuel_efficiency}")
        
        return ". ".join(content_parts)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for metadata storage."""
        return self.dict(exclude_none=True)


class TireCatalog(BaseModel):
    """Collection of tire products."""
    
    products: List[TireProduct] = Field(..., min_items=1, description="List of tire products")
    total_count: int = Field(..., ge=0, description="Total number of products")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
    
    @validator('total_count')
    def validate_total_count(cls, v, values):
        """Validate total count matches products length."""
        if 'products' in values and len(values['products']) != v:
            raise ValueError('Total count must match number of products')
        return v
    
    def get_by_id(self, product_id: str) -> Optional[TireProduct]:
        """Get product by ID."""
        for product in self.products:
            if product.id == product_id:
                return product
        return None
    
    def get_by_category(self, category: TireCategory) -> List[TireProduct]:
        """Get products by category."""
        return [p for p in self.products if p.category == category]
    
    def get_in_price_range(self, min_price: Decimal, max_price: Decimal) -> List[TireProduct]:
        """Get products in price range."""
        return [p for p in self.products if min_price <= p.price <= max_price]


class ChatMessage(BaseModel):
    """Chat message model."""
    
    role: str = Field(..., pattern=r"^(user|assistant|system)$", description="Message role")
    content: str = Field(..., min_length=1, max_length=4000, description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True


class ChatSession(BaseModel):
    """Chat session model."""
    
    session_id: str = Field(..., min_length=1, description="Session identifier")
    messages: List[ChatMessage] = Field(default_factory=list, description="Chat messages")
    created_at: Optional[str] = Field(None, description="Session creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    
    class Config:
        """Pydantic configuration."""
        validate_assignment = True
    
    def add_message(self, role: str, content: str, metadata: Optional[dict] = None) -> None:
        """Add a message to the session."""
        message = ChatMessage(role=role, content=content, metadata=metadata)
        self.messages.append(message)
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages."""
        return self.messages[-limit:] if limit > 0 else self.messages
    
    def clear_messages(self) -> None:
        """Clear all messages."""
        self.messages.clear()