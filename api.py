from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelInput(BaseModel):
    Car_Model: str
    Year: int
    Kms_Driven: int
    Suspension_Type: str
    Fuel_Type: str


# Load the saved model
Honda_model = joblib.load(open('Honda.pkl', 'rb'))


@app.post('/Honda_prediction')
def honda_prediction(input_parameters: ModelInput):
    try:
        # Convert input parameters to a list as expected by the model
        input_list = [
            input_parameters.Car_Model,
            input_parameters.Year,
            input_parameters.Kms_Driven,
            input_parameters.Suspension_Type,
            input_parameters.Fuel_Type
        ]

        print(input_list)
        prediction = Honda_model.predict([input_list])
        return {"prediction": int(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
