# Agentic Lender AI Engine

## Overview

This component provides an automated property valuation prototype for the
Agentic Lender system. It uses a scikit-learn regression model to estimate a
property's market value from its characteristics.

The API also compares the estimated market value with the asking price and
classifies the asking price as below, close to, or above the estimated value.

## Inputs

- County
- Floor area in square metres
- Number of bedrooms
- Property type
- BER rating
- Asking price

The asking price is used only for comparison. It is not used by the model when
predicting market value.

## Technologies

- Python
- pandas
- scikit-learn
- FastAPI
- Uvicorn
- joblib

## Project Structure

- `data/sample_irish_properties.csv` – synthetic property dataset
- `train_model.py` – trains and saves the valuation model
- `main.py` – provides the FastAPI endpoints
- `property_valuation_model.joblib` – trained model
- `requirements.txt` – Python dependencies

## Setup

Create and activate a virtual environment:

```powershell
py -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1