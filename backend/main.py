import os
import tempfile
import datetime
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from ml.disease_detector import predict_disease
from db_handler import save_disease_history, get_disease_history

app = FastAPI(title="AgriSaarthi API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent
CROP_DATA_PATH = BASE_DIR / "data" / "crop_profiles.csv"
PRICE_DATA_PATH = BASE_DIR / "data" / "market_prices.csv"

# Make sure data directory exists, otherwise use base dir
if not CROP_DATA_PATH.exists():
    CROP_DATA_PATH = BASE_DIR / "crop_profiles.csv"
if not PRICE_DATA_PATH.exists():
    PRICE_DATA_PATH = BASE_DIR / "market_prices.csv"

def load_crop_data():
    try:
        crops = pd.read_csv(CROP_DATA_PATH)
        prices = pd.read_csv(PRICE_DATA_PATH)

        crops.columns = crops.columns.str.strip().str.lower()
        prices.columns = prices.columns.str.strip().str.lower()

        if "crop" not in crops.columns and "crop_name" in crops.columns:
            crops = crops.rename(columns={"crop_name": "crop"})
        if "crop" not in prices.columns and "crop_name" in prices.columns:
            prices = prices.rename(columns={"crop_name": "crop"})

        return crops, prices
    except Exception as e:
        print(f"Error loading crop data: {e}")
        return pd.DataFrame(), pd.DataFrame()

crops, prices = load_crop_data()

class CropRequest(BaseModel):
    state: str
    climate_zone: str
    soil_ph: float
    water: str
    lang: str = "en"

@app.post("/api/recommend_crop")
def recommend_crop(req: CropRequest):
    if crops.empty or prices.empty:
        raise HTTPException(status_code=500, detail="Crop data not available")

    month = datetime.datetime.now().month
    season = ("Rainy / Monsoon" if month in [6, 7, 8, 9, 10]
              else "Winter" if month in [11, 12, 1, 2, 3]
              else "Summer")

    filtered = crops[
        (crops["ph_min"] <= req.soil_ph) &
        (crops["ph_max"] >= req.soil_ph) &
        (crops["water_need"].str.lower() == req.water.lower()) &
        (crops["climate_zone"].str.lower() == req.climate_zone.lower()) &
        (crops["season"].str.lower() == season.lower())
    ]

    if filtered.empty:
        return {"success": False, "message": "No matching crops found", "crops": []}

    result = pd.merge(filtered, prices, on="crop", how="inner")
    result["Profit_Index"] = result["base_yield"] * result["base_price"]
    result = result.sort_values("Profit_Index", ascending=False).head(3)

    recommended = []
    for _, row in result.iterrows():
        # Get localized name
        crop_name = row.get(f"crop_{req.lang}", row["crop"]) if req.lang in ["hi", "ta", "te", "ml"] else row["crop"]
        recommended.append({
            "crop": crop_name,
            "profit_index": float(row["Profit_Index"]),
            "water_need": row["water_need"],
            "carbon_footprint": row.get("carbon_footprint", "N/A"),
            "sowing_months": row.get("sowing_months", "N/A"),
            "fertilizer": row.get("fertilizer", "N/A")
        })

    return {"success": True, "crops": recommended}

@app.post("/api/detect_disease")
async def detect_disease(
    farmer_name: str = Form(...),
    lang: str = Form("en"),
    file: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predict_disease(tmp_path, lang=lang)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    # Resolve language specific text for display
    if lang in ["hi", "ta", "te", "ml"]:
        disease_display = result.get(f"disease_{lang}", result["disease"]).replace("_", " ")
        crop_display = result.get(f"crop_{lang}", result["crop"])
        remedy_display = result.get(f"remedy_{lang}", result["remedy"])
        precautions_display = result.get(f"precautions_{lang}", result["precautions"])
    else:
        disease_display = result["disease"].replace("_", " ")
        crop_display = result["crop"]
        remedy_display = result["remedy"]
        precautions_display = result["precautions"]

    save_disease_history(
        farmer_name, crop_display, disease_display,
        result.get("remedy_en", ""), result.get("precautions_en", ""),
        result.get("remedy_hi", ""), result.get("precautions_hi", ""),
        result.get("remedy_ta", ""), result.get("precautions_ta", ""),
        result.get("remedy_te", ""), result.get("precautions_te", ""),
        result.get("remedy_ml", ""), result.get("precautions_ml", ""),
    )

    return {
        "success": True,
        "disease": disease_display,
        "crop": crop_display,
        "remedy": remedy_display,
        "precautions": precautions_display,
        "confidence": result.get("confidence", 0)
    }

@app.get("/api/history")
def get_history():
    try:
        df = get_disease_history()
        # Convert df to list of dicts, handle NaN and NAT
        return {"success": True, "history": df.fillna("").to_dict(orient="records")}
    except Exception as e:
        return {"success": False, "message": str(e), "history": []}

@app.get("/")
def read_root():
    return {"status": "ok", "message": "AgriSaarthi API is running"}
