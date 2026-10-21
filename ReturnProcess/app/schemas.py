from pydantic import BaseModel, Field

class ShipmentRequest(BaseModel):
    tracking_number: str = Field(
        min_length=1,
        max_length=50
    )
    status: str = Field(
        min_length=1,
        max_length=100
    )
    latest_event: str = Field(
        min_length=1,
        max_length=500
    )
    last_location: str = Field(
        min_length=1,
        max_length=200
    )
    last_updated: str = Field(
        min_length=1,
        max_length=100
    )
class ShipmentExplanation(BaseModel):
    what_happened: str
    recommended_action: str
    unknowns: str

class ShipmentInsightsResponse(BaseModel):
    tracking_number: str
    status: str
    explanation: ShipmentExplanation

