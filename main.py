import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="Agentic Lender AI Engine",
    version="1.0"
)


# Allow the React application to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the model created by train_model.py
model = joblib.load("property_valuation_model.joblib")


class PropertyInput(BaseModel):
    county: str
    floor_area: float = Field(gt=0)
    bedrooms: int = Field(ge=1, le=10)
    property_type: str
    ber_rating: str
    asking_price: float = Field(gt=0)


@app.get("/health")
def health_check():
    return {
        "status": "online",
        "model_loaded": True
    }


@app.post("/predict")
def predict_property_value(property_data: PropertyInput):
    model_input = pd.DataFrame(
        [{
            "county": property_data.county,
            "floor_area": property_data.floor_area,
            "bedrooms": property_data.bedrooms,
            "property_type": property_data.property_type,
            "ber_rating": property_data.ber_rating
        }]
    )

    predicted_value = float(model.predict(model_input)[0])
    predicted_value = round(predicted_value, 2)

    difference = property_data.asking_price - predicted_value
    percentage_difference = (difference / predicted_value) * 100

    if percentage_difference > 5:
        price_assessment = "Above estimated market value"
    elif percentage_difference < -5:
        price_assessment = "Below estimated market value"
    else:
        price_assessment = "Close to estimated market value"

    return {
        "estimated_value": predicted_value,
        "asking_price": property_data.asking_price,
        "difference": round(difference, 2),
        "percentage_difference": round(percentage_difference, 2),
        "price_assessment": price_assessment
    }
