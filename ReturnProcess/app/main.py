import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.schemas import (
    ShipmentRequest,
    ShipmentExplanation,
    ShipmentInsightsResponse
)
from app.shipment_data import SHIPMENT
from app.ai_service import explain_shipment

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Shipment Exception API",
    description=(
        "Explains shipment tracking events using "
        "LangChain and an LLM."
    ),
    version="1.0.0"
)

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "shipment-ai-api"
    }

@app.get(
    "/shipments/{tracking_number}/insights",
    response_model=ShipmentInsightsResponse
)

def get_shipment_insights(tracking_number: str):

    # Find the shipment in our local mock data.
    shipment = SHIPMENT.get(tracking_number)

    if shipment is None:
        raise HTTPException(
            status_code=404,
            detail="Shipment not found"
        )

    try:
        # Pass the shipment facts to LangChain.
        explanation = explain_shipment(shipment)

        return ShipmentInsightsResponse(
            tracking_number=shipment["tracking_number"],
            status=shipment["status"],
            explanation=explanation
        )

    except Exception:
        # Log the actual error securely in a production app.
        raise HTTPException(
            status_code=502,
            detail="Unable to generate shipment insights"
        )


@app.post(
    "/explain",
    response_model=ShipmentInsightsResponse
)
def explain_shipment_endpoint(
    shipment: ShipmentRequest
):
    try:
        # .NET can send its already-authorized shipment data.
        shipment_data = shipment.model_dump()

        explanation = explain_shipment(shipment_data)

        return ShipmentInsightsResponse(
            tracking_number=shipment.tracking_number,
            status=shipment.status,
            explanation=explanation
        )

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="Unable to generate shipment insights"
        )
