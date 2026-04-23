<div align="center">

# 🌾 AgriSaarthi

### AI-Powered Crop & Disease Assistant for Indian Farmers

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-ML%20Model-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

*"Every farmer deserves a Saarthi."*

</div>

---

## 📖 What is AgriSaarthi?

**AgriSaarthi** (*Saarthi* = guide in Hindi) is a multilingual, AI-powered Streamlit web app that helps Indian farmers make smarter decisions — without needing to know English or have technical expertise.

It combines machine learning, crop science, and localized data to offer three core tools in one dashboard:

| Module | What it does |
|---|---|
| 🌾 **Crop Recommendation** | Suggests the most profitable crops for your state, soil, and season |
| 📸 **Disease Detection** | Diagnose plant diseases from a leaf photo using a trained CNN model |
| 📜 **Farmer History** | Track past disease diagnoses per farmer |

---

## 🌐 Language Support

AgriSaarthi is fully localized in **5 Indian languages**, automatically selected based on the farmer's state:

| Language | Script | Auto-selected for |
|---|---|---|
| English | Latin | Karnataka, Goa, Maharashtra, West Bengal, Odisha |
| हिंदी | Devanagari | Punjab, Haryana, UP, MP, Bihar, Rajasthan, Gujarat, J&K |
| தமிழ் | Tamil | Tamil Nadu |
| తెలుగు | Telugu | Andhra Pradesh |
| മലയാളം | Malayalam | Kerala |

Farmers can also manually override the language at any time.

---

## 🧠 Features In Depth

### 🌾 Crop Recommendation Engine

- Select your **Indian state** → climate zone is auto-inferred (Tropical / Temperate / Dry)
- Input **soil pH** (slider: 4.5–9.0) and **water availability**
- Current **season is auto-detected** from the system date (Monsoon / Winter / Summer)
- Crops are filtered from `crop_profiles.csv` against all constraints
- Top 3 crops are ranked by a **Profit Index** (`base_yield × base_price`)
- Results show: crop name (in local language), profit index, water need, carbon footprint, sowing months, and fertilizer recommendations
- Interactive **bar chart** of profit index rendered via Streamlit

### 📸 Plant Disease Detector

- Farmer enters their **name** (for history tracking)
- **Upload a leaf photo** (JPG/PNG)
- A trained **Keras CNN model** (`plant_disease_model.h5`) runs inference on the image
- Results displayed in the farmer's language:
  - Disease name
  - Affected crop
  - Remedy / treatment
  - Precautions
  - Confidence score (%)
- Diagnosis is **saved to `disease_history.csv`** with multilingual remedy data for all 5 languages

### 📜 Farmer History

- View a full log of all past disease detections across all farmers
- Tabular view with disease, crop, and remedy columns

---

## 🗂️ Project Structure

```
AgriSaarthi/
│
├── app.py                      # Main Streamlit application
│
├── ml/
│   ├── disease_detector.py     # CNN inference pipeline (Keras)
│   └── crop_recommender.py     # Crop filtering & profit ranking logic
│
├── data/
│   ├── crop_profiles.csv       # Crop database (pH range, water, season, zone, yield)
│   ├── market_prices.csv       # Commodity price data per crop
│   ├── disease_remedies.json   # Disease → remedy mapping (5 languages)
│   ├── disease_history.csv     # Farmer diagnosis history (auto-generated)
│   └── uploaded_leaf.jpg       # Temp file for uploaded leaf images
│
├── models/
│   └── plant_disease_model.h5  # Trained Keras CNN for plant disease classification
│
└── farmer_history.db           # SQLite DB (farmer records)
```

---

## ⚙️ Tech Stack

| Component | Technology |
|---|---|
| **Web Framework** | Streamlit |
| **ML / Inference** | TensorFlow / Keras (`.h5` CNN model) |
| **Data Processing** | Pandas |
| **Disease Remedies** | JSON (multilingual) |
| **History Storage** | CSV + SQLite |
| **Language** | Python 3.9+ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/1HPdhruv/AgriSaarthi.git
cd AgriSaarthi

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note:** If a `requirements.txt` is not present, install manually:
> ```bash
> pip install streamlit pandas tensorflow pillow
> ```

### Run the App

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📊 Data Files

| File | Description |
|---|---|
| `crop_profiles.csv` | Each row is a crop with pH range, water need, climate zone, season, base yield, carbon footprint, sowing months, fertilizer |
| `market_prices.csv` | Base price per crop used for profit index calculation |
| `disease_remedies.json` | Maps disease class names to remedies and precautions in all 5 languages |
| `disease_history.csv` | Auto-created on first diagnosis; stores per-farmer disease history |

---

## 🗺️ State → Climate Zone Mapping

| Zone | States |
|---|---|
| **Tropical** | Kerala, Karnataka, Tamil Nadu, Andhra Pradesh, Goa, Maharashtra, West Bengal, Odisha |
| **Temperate** | Punjab, Haryana, Uttar Pradesh, Madhya Pradesh, Bihar, Himachal Pradesh, J&K |
| **Dry / Arid** | Rajasthan, Gujarat, Ladakh |

---

## 🔭 Roadmap

- [ ] Add more crop entries across all climate zones and seasons
- [ ] Retrain disease model on a larger, more diverse dataset
- [ ] Add weather API integration for real-time irrigation advice
- [ ] Government scheme lookup by state
- [ ] Voice input for farmers with low literacy
- [ ] Mobile-optimized UI
- [ ] Offline mode for low-connectivity rural areas

---

## 🏆 Recognition

AgriSaarthi was built for the [**World's Largest Hackathon presented by Bolt**](https://worldslargesthackathon.devpost.com/).

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch: `git checkout -b feature/your-feature`
3. Commit: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👨‍💻 Team

Built with ❤️ for Indian farmers.

| Name | GitHub |
|---|---|
| Dhruv | [@1HPdhruv](https://github.com/1HPdhruv) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
