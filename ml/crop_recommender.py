import random

# Hardcoded sample crops (you can expand this list)
CROPS = [
    "Wheat", "Rice", "Maize", "Tomato", "Cotton",
    "Sugarcane", "Soybean", "Potato", "Mustard"
]

def recommend_crop(soil_type: str, rainfall: float, temperature: float) -> str:
    """
    Mock crop recommendation system.
    - Ignores actual ML model (since .pkl is empty).
    - Uses simple logic + randomness to pick a crop.
    """

    # Very simple rule-based filtering
    if "clay" in soil_type.lower():
        candidates = ["Rice", "Sugarcane"]
    elif "sandy" in soil_type.lower():
        candidates = ["Maize", "Cotton"]
    else:
        candidates = CROPS  # fallback

    # Temperature & rainfall adjustment
    if temperature < 20:
        candidates = [c for c in candidates if c not in ["Cotton", "Sugarcane"]]
    if rainfall < 500:
        candidates = [c for c in candidates if c not in ["Rice", "Sugarcane"]]

    # Pick one deterministically (for demo consistency)
    if not candidates:
        candidates = CROPS
    return random.choice(candidates)

# ✅ Quick test
if __name__ == "__main__":
    print("🌱 Recommended crop:", recommend_crop("clay soil", 800, 25))
    print("🌱 Recommended crop:", recommend_crop("sandy soil", 300, 30))
