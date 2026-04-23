import streamlit as st
import os
import pandas as pd
from ml.disease_detector import predict_disease

# -------------------------
# 📌 Load Data
# -------------------------
CROP_DATA_PATH = "data/crop_profiles.csv"
PRICE_DATA_PATH = "data/market_prices1.csv"

@st.cache_data
def load_crop_data():
    crops = pd.read_csv(CROP_DATA_PATH)
    prices = pd.read_csv(PRICE_DATA_PATH)

    # ✅ Normalize column names (lowercase, strip spaces)
    crops.columns = crops.columns.str.strip().str.lower()
    prices.columns = prices.columns.str.strip().str.lower()

    # ✅ Ensure both have "crop" column
    if "crop" not in crops.columns:
        if "crop_name" in crops.columns:
            crops = crops.rename(columns={"crop_name": "crop"})
    if "crop" not in prices.columns:
        if "crop_name" in prices.columns:
            prices = prices.rename(columns={"crop_name": "crop"})

    return crops, prices

crops, prices = load_crop_data()

# -------------------------
# Dummy disease history saver (CSV-based for 5 languages)
# -------------------------
def save_disease_history(
    farmer, crop, disease,
    remedy_en, precautions_en,
    remedy_hi, precautions_hi,
    remedy_ta, precautions_ta,
    remedy_te, precautions_te,
    remedy_ml, precautions_ml
):
    """Mock history saver (writes to CSV)."""
    history_path = "data/disease_history.csv"
    entry = pd.DataFrame([{
        "farmer": farmer,
        "crop": crop,
        "disease": disease,
        "remedy_en": remedy_en, "precautions_en": precautions_en,
        "remedy_hi": remedy_hi, "precautions_hi": precautions_hi,
        "remedy_ta": remedy_ta, "precautions_ta": precautions_ta,
        "remedy_te": remedy_te, "precautions_te": precautions_te,
        "remedy_ml": remedy_ml, "precautions_ml": precautions_ml
    }])

    if os.path.exists(history_path):
        entry.to_csv(history_path, mode="a", header=False, index=False)
    else:
        entry.to_csv(history_path, index=False)

# -------------------------
# 📌 Streamlit UI
# -------------------------
st.set_page_config(page_title="AgriSaarthi", layout="wide")
st.title("🌱 AgriSaarthi - AI Crop & Disease Assistant")

tab1, tab2, tab3 = st.tabs(["🌾 Crop Recommendation", "📸 Disease Detection", "📜 Farmer History"])
# Map Indian states to climate zones
state_to_zone = {
    # Tropical
    "Kerala": "Tropical", "Karnataka": "Tropical", "Tamil Nadu": "Tropical",
    "Andhra Pradesh": "Tropical", "Goa": "Tropical", "Maharashtra": "Tropical",
    "West Bengal": "Tropical", "Odisha": "Tropical",

    # Temperate / Subtropical
    "Punjab": "Temperate", "Haryana": "Temperate", "Uttar Pradesh": "Temperate",
    "Madhya Pradesh": "Temperate", "Bihar": "Temperate", "Himachal Pradesh": "Temperate",
    "Jammu & Kashmir": "Temperate",

    # Dry / Arid
    "Rajasthan": "Dry", "Gujarat": "Dry", "Ladakh": "Dry"
}
# -------------------------
# 🌾 Crop Recommendation Tab
# -------------------------
with tab1:
    import datetime

    # -------------------
    # Map Indian states to climate zones
    # -------------------
    state_to_zone = {
        "Kerala": "Tropical", "Karnataka": "Tropical", "Tamil Nadu": "Tropical",
        "Andhra Pradesh": "Tropical", "Goa": "Tropical", "Maharashtra": "Tropical",
        "West Bengal": "Tropical", "Odisha": "Tropical",
        "Punjab": "Temperate", "Haryana": "Temperate", "Uttar Pradesh": "Temperate",
        "Madhya Pradesh": "Temperate", "Bihar": "Temperate", "Himachal Pradesh": "Temperate",
        "Jammu & Kashmir": "Temperate",
        "Rajasthan": "Dry", "Gujarat": "Dry", "Ladakh": "Dry"
    }

    # -------------------
    # Map Indian states to language
    # -------------------
    state_to_language = {
        "Punjab": "hi", "Haryana": "hi", "Uttar Pradesh": "hi", "Madhya Pradesh": "hi",
        "Bihar": "hi", "Himachal Pradesh": "hi", "Jammu & Kashmir": "hi", "Rajasthan": "hi", "Gujarat": "hi",
        "Kerala": "ml", "Karnataka": "en", "Tamil Nadu": "ta", "Andhra Pradesh": "te",
        "Goa": "en", "Maharashtra": "en", "West Bengal": "en", "Odisha": "en", "Ladakh": "en"
    }

    # -------------------
    # Multilingual labels
    # -------------------
    lang_dict = {
        "en": {"crop_header":"🌾 Crop Recommendation System","state":"Select State","soil_ph":"Soil pH",
               "water":"Water Availability","recommend_btn":"Recommend Crop","recommended_crops":"Recommended Crops:",
               "profit_chart":"Profit Index Chart","profit_values":"Profit Index Values:",
               "no_match":"No matching crops found for the given criteria.","water_labels":{"Low":"Low","Medium":"Medium","High":"High","Very High":"Very High"},
               "sowing":"Sowing Months","fertilizer":"Fertilizer","profit_index":"Profit Index","carbon":"Carbon Footprint"},
        "hi": {"crop_header":"🌾 फ़सल सिफारिश प्रणाली","state":"राज्य चुनें","soil_ph":"मिट्टी का pH",
               "water":"पानी की उपलब्धता","recommend_btn":"फसल सुझाएँ","recommended_crops":"अनुशंसित फ़सलें:",
               "profit_chart":"लाभ सूचकांक चार्ट","profit_values":"लाभ सूची",
               "no_match":"दिए गए मापदंडों के लिए कोई उपयुक्त फ़सल नहीं मिली।","water_labels":{"Low":"कम","Medium":"मध्यम","High":"अधिक","Very High":"बहुत अधिक"},
               "sowing":"बोआई का समय","fertilizer":"उर्वरक","profit_index":"लाभ सूचकांक","carbon":"कार्बन पदचिह्न"},
        "ta": {"crop_header":"🌾 பயிர் பரிந்துரை அமைப்பு","state":"மாநிலம் தேர்ந்தெடுக்கவும்","soil_ph":"மண்ணின் pH",
               "water":"நீர Availability","recommend_btn":"பயிர் பரிந்துரை செய்யவும்","recommended_crops":"பரிந்துரைக்கப்பட்ட பயிர்கள்:",
               "profit_chart":"பயன் குறியீட்டு வரைபடம்","profit_values":"பயன் மதிப்புகள்:",
               "no_match":"காணப்படும் பொருந்தக்கூடிய பயிர்கள் இல்லை.","water_labels":{"Low":"குறைந்த","Medium":"மதியம்","High":"அதிக","Very High":"மிக அதிக"},
               "sowing":"நட்டும் மாதங்கள்","fertilizer":"உரம்","profit_index":"பயன் குறியீடு","carbon":"கார்பன் பாதிப்பு"},
        "te": {"crop_header":"🌾 పంట సిఫారసుల వ్యవస్థ","state":"రాజ్యాన్ని ఎంచుకోండి","soil_ph":"మట్టిపీహెచ్",
               "water":"నీటి అందుబాటు","recommend_btn":"పంటను సిఫారసు చేయండి","recommended_crops":"సిఫార్సు చేసిన పంటలు:",
               "profit_chart":"లాభ సూచిక చార్ట్","profit_values":"లాభ సూచిక విలువలు:",
               "no_match":"ఇచ్చిన ప్రమాణాలకు సరిపడే పంటలు లభించలేదు.","water_labels":{"Low":"తక్కువ","Medium":"మధ్యస్థ","High":"ఎక్కువ","Very High":"చాలా ఎక్కువ"},
               "sowing":"నాటే నెలలు","fertilizer":"రసాయన పుష్కరణ","profit_index":"లాభ సూచిక","carbon":"కార్బన్ పాదచిహ్నం"},
        "ml": {"crop_header":"🌾 ഫസൽ ശിപാർശാ സംവിധാനം","state":"സംസ്ഥാനം തിരഞ്ഞെടുക്കുക","soil_ph":"മണ്ണിന്റെ pH",
               "water":"ജല ലഭ്യത","recommend_btn":"ഫസൽ ശിപാർശ ചെയ്യുക","recommended_crops":"ശിപാർശ ചെയ്ത ഫസലുകൾ:",
               "profit_chart":"ലാഭ സൂചിക ചാർട്ട്","profit_values":"ലാഭ സൂചിക മൂല്യങ്ങൾ:",
               "no_match":"നൽകിയ മാനദണ്ഡങ്ങൾക്ക് യോജിക്കുന്ന ഫസലുകൾ ഒന്നും കണ്ടെത്തിയില്ല.","water_labels":{"Low":"കുറഞ്ഞത്","Medium":"മധ്യമം","High":"അധികം","Very High":"വളരെ അധികം"},
               "sowing":"നട്ടു മാസങ്ങൾ","fertilizer":"ഉർവരം","profit_index":"ലാഭ സൂചിക","carbon":"കാർബൺ ഫുട്പ്രിന്റ്"}
    }

    # -------------------
    # Session state to track previous state & language
    # -------------------
    if "prev_state" not in st.session_state:
        st.session_state.prev_state = None
    if "lang_override" not in st.session_state:
        st.session_state.lang_override = "en"

    # -------------------
    # State selection
    # -------------------
    state = st.selectbox("State", list(state_to_zone.keys()), key="state_select")
    climate_zone = state_to_zone[state]

    # Reset language override if state changed
    if st.session_state.prev_state != state:
        st.session_state.prev_state = state
        st.session_state.lang_override = state_to_language.get(state, "en")

    # -------------------
    # Language override radio
    # -------------------
    target_lang = st.radio(
        "Language / भाषा / மொழி / భాష / ഭാഷ",
        options=["en","hi","ta","te","ml"],
        index=["en","hi","ta","te","ml"].index(st.session_state.lang_override),
        format_func=lambda x: {"en":"English","hi":"हिंदी","ta":"தமிழ்","te":"తెలుగు","ml":"മലയാളം"}[x],
        horizontal=True,
        key="lang_radio"
    )

    # Update session state if user manually changes radio
    st.session_state.lang_override = target_lang
    t = lang_dict[target_lang]

    st.header(t["crop_header"])

    # -------------------
    # Farmer inputs
    # -------------------
    soil_ph = st.slider("🌱 "+t["soil_ph"],4.5,9.0,6.5)
    water_options = list(t["water_labels"].values()) if target_lang in ["hi","ta","te","ml"] else list(t["water_labels"].keys())
    water = st.selectbox("💧 "+t["water"], water_options)

    water_map = {
        "कम":"Low","मध्यम":"Medium","अधिक":"High","बहुत अधिक":"Very High",
        "குறைந்த":"Low","மதியம்":"Medium","அதிக":"High","மிக அதிக":"Very High",
        "తక్కువ":"Low","మధ్యస్థ":"Medium","ఎక్కువ":"High","చాలా ఎక్కువ":"Very High",
        "കുറഞ്ഞത്":"Low","മധ്യമം":"Medium","അധികം":"High","വളരെ അധികം":"Very High",
        "Low":"Low","Medium":"Medium","High":"High","Very High":"Very High"
    }
    water_value = water_map.get(water, water)

    # -------------------
    # Season detection
    # -------------------
    month = datetime.datetime.now().month
    season = "Rainy / Monsoon" if month in [6,7,8,9,10] else "Winter" if month in [11,12,1,2,3] else "Summer"

    # -------------------
    # Recommend Crop
    # -------------------
    if st.button("✅ "+t["recommend_btn"]):
        filtered = crops[
            (crops["ph_min"]<=soil_ph)&
            (crops["ph_max"]>=soil_ph)&
            (crops["water_need"].str.lower()==water_value.lower())&
            (crops["climate_zone"].str.lower()==climate_zone.lower())&
            (crops["season"].str.lower()==season.lower())
        ]

        if not filtered.empty:
            result = pd.merge(filtered, prices, on="crop", how="inner")
            result["Profit_Index"] = result["base_yield"]*result["base_price"]
            result = result.sort_values(by="Profit_Index", ascending=False).head(3)

            st.success("✅ "+t["recommended_crops"])

            for _, row in result.iterrows():
                if target_lang=="hi": crop_name=row.get("crop_hi",row["crop"])
                elif target_lang=="ta": crop_name=row.get("crop_ta",row["crop"])
                elif target_lang=="te": crop_name=row.get("crop_te",row["crop"])
                elif target_lang=="ml": crop_name=row.get("crop_ml",row["crop"])
                else: crop_name=row["crop"]

                water_label=t["water_labels"].get(row["water_need"],row["water_need"])
                st.write(f"**{crop_name}**")
                st.write(f"💰 {t['profit_index']}: {row['Profit_Index']}")
                st.write(f"💧 {t['water']}: {water_label}")
                st.write(f"🌍 {t['carbon']}: {row['carbon_footprint']}")
                st.write(f"🗓️ {t['sowing']}: {row.get('sowing_months','N/A')}")
                st.write(f"🧪 {t['fertilizer']}: {row.get('fertilizer','N/A')}")
                st.markdown("---")

            # Profit Chart
            chart_labels = [row.get(f"crop_{target_lang}", row["crop"]) if target_lang in ["hi","ta","te","ml"] else row["crop"] for _, row in result.iterrows()]
            chart_data = pd.DataFrame({'Crop': chart_labels,'Profit_Index':result['Profit_Index']})
            chart_data = chart_data.set_index('Crop').sort_values(by='Profit_Index',ascending=False)
            st.subheader("📊 "+t["profit_chart"])
            st.bar_chart(chart_data)
            st.write("💰 "+t["profit_values"])
            st.table(chart_data)

        else:
            st.error("⚠️ "+t["no_match"])

# -------------------------
# 📸 Disease Detection Tab
# -------------------------
with tab2:
    st.header("📸 Plant Disease Detection")

    # Farmer name input
    farmer_name = st.text_input(
        "Farmer Name / किसान का नाम / விவசாயி பெயர் / రైతు పేరు / ഫാർമർ പേര്"
    )
    if not farmer_name:
        st.warning("Please enter your name to proceed.")
        st.stop()

    # Language selection
    lang = st.radio(
        "Choose language / भाषा / மொழி / భాష / ഭാഷ:",
        ["en", "hi", "ta", "te", "ml"],
        format_func=lambda x: {
            "en": "English",
            "hi": "हिंदी",
            "ta": "தமிழ்",
            "te": "తెలుగు",
            "ml": "മലയാളം"
        }[x]
    )

    # Leaf image uploader
    uploaded_file = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        img_path = os.path.join("data", "uploaded_leaf.jpg")
        with open(img_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(img_path, caption="Uploaded Leaf", use_container_width=True)

        with st.spinner("🔍 Analyzing leaf image..."):
            from ml.disease_detector import predict_disease  # adjust import path if needed
            result = predict_disease(img_path, lang=lang)

        st.success("✅ Prediction Complete!")

        # -------------------
        # Language-based display
        # -------------------
        if lang == "hi":
            disease_display = result.get("disease_hi", result["disease"]).replace("_", " ")
            crop_display = result.get("crop_hi", result["crop"])
            remedy_display = result.get("remedy_hi", result["remedy"])
            precautions_display = result.get("precautions_hi", result["precautions"])
        elif lang == "ta":
            disease_display = result.get("disease_ta", result["disease"]).replace("_", " ")
            crop_display = result.get("crop_ta", result["crop"])
            remedy_display = result.get("remedy_ta", result["remedy"])
            precautions_display = result.get("precautions_ta", result["precautions"])
        elif lang == "te":
            disease_display = result.get("disease_te", result["disease"]).replace("_", " ")
            crop_display = result.get("crop_te", result["crop"])
            remedy_display = result.get("remedy_te", result["remedy"])
            precautions_display = result.get("precautions_te", result["precautions"])
        elif lang == "ml":
            disease_display = result.get("disease_ml", result["disease"]).replace("_", " ")
            crop_display = result.get("crop_ml", result["crop"])
            remedy_display = result.get("remedy_ml", result["remedy"])
            precautions_display = result.get("precautions_ml", result["precautions"])
        else:
            # default English
            disease_display = result["disease"].replace("_", " ")
            crop_display = result["crop"]
            remedy_display = result["remedy"]
            precautions_display = result["precautions"]

        # -------------------
        # Display results
        # -------------------
        st.subheader(f"Disease: {disease_display}")
        st.write(f"🌾 Crop: **{crop_display}**")
        st.write(f"💊 Remedy: {remedy_display}")
        st.write(f"⚠️ Precautions: {precautions_display}")
        st.write(f"🔎 Confidence: {result.get('confidence', 0)*100:.2f}%")

        # -------------------
        # Save to history
        # -------------------
        save_disease_history(
            farmer_name,
            crop_display,
            disease_display,
            result.get("remedy_en", ""),
            result.get("precautions_en", ""),
            result.get("remedy_hi", ""),
            result.get("precautions_hi", ""),
            result.get("remedy_ta", ""),
            result.get("precautions_ta", ""),
            result.get("remedy_te", ""),
            result.get("precautions_te", ""),
            result.get("remedy_ml", ""),
            result.get("precautions_ml", "")
        )
        st.info("📌 Saved to history!")
# -------------------------
# 📜 Farmer History Tab
# -------------------------
with tab3:
    st.header("📜 Farmer Disease History")

    def get_disease_history():
        history_path = "data/disease_history.csv"
        if os.path.exists(history_path):
            df = pd.read_csv(history_path)
            return df
        else:
            return pd.DataFrame()

    history_df = get_disease_history()

    if history_df.empty:
        st.info("No history found yet.")
    else:
        st.dataframe(history_df)