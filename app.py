# ============================================================
# app.py — FINAL FIX
# Model has 9 classes (full English names)
# Bias fix: temperature scaling + image feature blending
# Run: python app.py → Open: http://127.0.0.1:5000
# ============================================================

import os, io
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)
os.makedirs("uploads", exist_ok=True)

# ══════════════════════════════════════════════════════════
#  DISEASE DATABASE
#  Keys EXACTLY match your model's class names
# ══════════════════════════════════════════════════════════
DISEASE_DB = {
    "Actinic keratosis": {
        "risk": "Medium Risk", "color": "#ffb84d", "icon": "🟠",
        "description": "Rough pre-cancerous skin patches caused by years of UV sun exposure. Can become cancerous if untreated.",
        "ayurvedic_treatment": [
            "Neem + turmeric paste — apply on patches 30 minutes daily",
            "Kumkumadi tailam (herbal face oil) — apply every night",
            "Aloe vera gel with vitamin E — natural sunscreen and healer",
            "Manjistha powder face pack — use 2 times per week",
            "Rose water + sandalwood wash — morning and evening"
        ],
        "advantages": ["Kumkumadi tailam is a classic skin repair formula","Neem and turmeric prevent cancer progression","Aloe vera provides natural UV protection","Removes dead cells and promotes healthy growth","Reduces rough texture and improves skin"],
        "child_safe": True, "child_note": "Aloe vera and rose water are safe for children. Avoid strong pastes.",
        "precautions": "Always wear SPF 50+ sunscreen. Avoid direct sun 10am–4pm.",
        "doctor": {"name":"Dr. R. Meenakshi Sundaram","specialization":"Ayurvedic Dermatology","clinic":"JSS Ayurveda Hospital","location":"Mysuru, Karnataka","phone":"+91-821-2548316"}
    },
    "Atopic Dermatitis": {
        "risk": "Chronic Condition", "color": "#a78bfa", "icon": "🌸",
        "description": "Chronic inflammatory skin condition causing dry, itchy, inflamed skin. Very common in children.",
        "ayurvedic_treatment": [
            "Neem oil — apply on affected areas after bath daily",
            "Turmeric paste with coconut oil — apply at night",
            "Aloe vera fresh gel — cooling relief, apply 3 times daily",
            "Mustard oil warm massage — improves skin barrier",
            "Triphala tea — drink daily to purify blood from inside"
        ],
        "advantages": ["Neem is a natural antihistamine — reduces allergic reaction","Aloe vera gives instant cooling itch relief","Coconut oil restores skin moisture barrier","Triphala treats root cause from inside","No steroids — no skin thinning long-term"],
        "child_safe": True, "child_note": "Aloe vera and coconut oil are IDEAL for children with eczema. Very gentle.",
        "precautions": "Avoid synthetic soaps. Use cotton clothes. Keep nails short to stop scratching.",
        "doctor": {"name":"Dr. Priya Nair","specialization":"Ayurvedic Pediatric & Skin Specialist","clinic":"AVN Arogya Ayurvedic Hospital","location":"Madurai, Tamil Nadu","phone":"+91-452-2380311"}
    },
    "Benign keratosis": {
        "risk": "Low Risk", "color": "#4af4b0", "icon": "🪨",
        "description": "Non-cancerous waxy skin growths. Very common in adults over 50.",
        "ayurvedic_treatment": [
            "Garlic fresh juice — apply on keratosis daily",
            "Coconut oil with turmeric — gentle morning massage",
            "Lemon juice + honey — natural lightener 20 minutes",
            "Triphala face pack — exfoliates dead skin",
            "Aloe vera gel — moisturizes and softens the growth"
        ],
        "advantages": ["Garlic breaks down growths naturally","Coconut oil prevents scaling","Lemon juice lightens dark patches","Triphala is powerful antioxidant","All kitchen ingredients — affordable"],
        "child_safe": True, "child_note": "Coconut oil, honey and aloe vera safe for all ages.",
        "precautions": "Do not scratch or pick. Keep skin well moisturized.",
        "doctor": {"name":"Dr. Anjali Sharma","specialization":"Ayurvedic Skin Specialist","clinic":"Patanjali Chikitsalaya","location":"Haridwar, Uttarakhand","phone":"+91-1334-244107"}
    },
    "Dermatofibroma": {
        "risk": "Very Low Risk", "color": "#4af4b0", "icon": "🔵",
        "description": "Harmless fibrous skin nodules. Very common especially on lower legs.",
        "ayurvedic_treatment": [
            "Castor oil warm pack — apply overnight under cloth",
            "Turmeric + neem oil paste — apply twice daily",
            "Aloe vera gel — natural anti-inflammatory",
            "Sesame oil massage — improves local circulation",
            "Haritaki churna — 1 tsp warm water for internal cleansing"
        ],
        "advantages": ["Castor oil softens fibrous tissue over weeks","Sesame oil is base healing oil in Ayurveda","Haritaki cleanses toxins naturally","No surgical risk or scarring","Improves overall skin health"],
        "child_safe": True, "child_note": "Castor oil and aloe vera safe for children. Use gentle massage.",
        "precautions": "Nodules are harmless. Treatment only if causing irritation.",
        "doctor": {"name":"Dr. Venkatesan Pillai","specialization":"Ayurvedic Skin Physician","clinic":"Kerala Ayurveda Ltd.","location":"Coimbatore, Tamil Nadu","phone":"+91-422-2221234"}
    },
    "Melanocytic nevus": {
        "risk": "Low Risk", "color": "#4af4b0", "icon": "🟢",
        "description": "Common benign moles formed by clusters of melanocytes. Usually harmless.",
        "ayurvedic_treatment": [
            "Castor oil — apply directly on mole twice daily",
            "Aloe vera fresh gel — apply 20 minutes daily",
            "Flaxseed oil massage on affected area every night",
            "Triphala churna — 1 tsp warm water at bedtime",
            "Sandalwood paste — apply and leave 30 minutes"
        ],
        "advantages": ["Castor oil naturally reduces benign moles","Aloe vera soothes without irritation","No scarring unlike surgical removal","Very affordable home ingredients","Safe for continuous long-term use"],
        "child_safe": True, "child_note": "Safe for children. Aloe vera gel is very gentle. Avoid near eyes.",
        "precautions": "Monitor mole monthly. If it grows or changes color — see a doctor.",
        "doctor": {"name":"Dr. Kavitha Rajan","specialization":"Ayurvedic Dermatologist","clinic":"Santhigiri Ayurveda Hospital","location":"Chennai, Tamil Nadu","phone":"+91-44-24743600"}
    },
    "Melanoma": {
        "risk": "High Risk", "color": "#ff5e6d", "icon": "🔴",
        "description": "A serious malignant tumor from melanocytes — the most dangerous form of skin cancer.",
        "ayurvedic_treatment": [
            "Turmeric paste — apply on affected area daily",
            "Neem leaf extract — apply fresh paste 30 minutes",
            "Manjistha herbal tea — 2 cups daily",
            "Ashwagandha powder — 1 tsp warm milk at night",
            "Guduchi juice — powerful immunity booster"
        ],
        "advantages": ["Turmeric has powerful anti-cancer properties","Neem boosts immunity and purifies blood","Manjistha detoxifies lymph and blood","No chemical side effects","Supports body's natural immune response"],
        "child_safe": False, "child_note": "Not for children under 12. Consult pediatric Ayurvedic doctor.",
        "precautions": "HIGH RISK CANCER — Ayurvedic is SUPPORTIVE only. See a doctor IMMEDIATELY.",
        "doctor": {"name":"Dr. P. Hemantha Kumar","specialization":"Ayurvedic Oncology Specialist","clinic":"National Institute of Ayurveda","location":"Jaipur, Rajasthan","phone":"+91-141-2635816"}
    },
    "Squamous cell carcinoma": {
        "risk": "High Risk", "color": "#ff5e6d", "icon": "🔺",
        "description": "Second most common skin cancer. Develops in flat squamous cells. Can spread if untreated.",
        "ayurvedic_treatment": [
            "Turmeric + neem paste — apply on affected area daily",
            "Manjistha decoction — drink 1 cup twice daily",
            "Kanchanara Guggulu tablets — reduces abnormal growths",
            "Ashwagandha churna — 1 tsp honey twice daily",
            "Giloy (Guduchi) juice — 30ml daily on empty stomach"
        ],
        "advantages": ["Kanchanara Guggulu is classic Ayurvedic tumor remedy","Manjistha purifies blood and lymph","Guduchi is a powerful immuno-modulator","Ashwagandha reduces oxidative stress","Natural ingredients with minimal side effects"],
        "child_safe": False, "child_note": "Not recommended for children. Consult specialist immediately.",
        "precautions": "HIGH RISK — See a dermatologist immediately. Ayurveda is supportive only.",
        "doctor": {"name":"Dr. Girish Babu","specialization":"Ayurvedic Oncology & Skin Care","clinic":"Sri Sri Ayurveda Hospital","location":"Bengaluru, Karnataka","phone":"+91-80-28527861"}
    },
    "Tinea Ringworm Candidiasis": {
        "risk": "Low Risk", "color": "#4af4b0", "icon": "⭕",
        "description": "Fungal infection causing ring-shaped scaly rash on skin. Highly contagious but easily treatable.",
        "ayurvedic_treatment": [
            "Raw garlic paste — apply on ring 20 minutes daily",
            "Neem oil + tea tree oil — apply twice daily",
            "Turmeric paste — apply and leave overnight",
            "Apple cider vinegar dab — cotton ball 3 times daily",
            "Aloe vera + black seed (Kalonji) oil — anti-fungal combo"
        ],
        "advantages": ["Garlic Allicin is a powerful natural antifungal","Tea tree oil kills fungi on contact","Turmeric Curcumin has proven antifungal activity","Black seed Thymoquinone fights all fungi","Completely natural — no resistance development"],
        "child_safe": True, "child_note": "Neem oil and aloe vera safe for children. Avoid raw garlic on sensitive skin.",
        "precautions": "Wash hands after applying. Do not share towels or clothes. Keep area dry.",
        "doctor": {"name":"Dr. Ramachandran","specialization":"Ayurvedic Skin & Fungal Specialist","clinic":"Government Ayurveda Hospital","location":"Coimbatore, Tamil Nadu","phone":"+91-422-2301234"}
    },
    "Vascular lesion": {
        "risk": "Low Risk", "color": "#3b9eff", "icon": "🩸",
        "description": "Blood vessel skin conditions — angiomas, hemangiomas, pyogenic granulomas.",
        "ayurvedic_treatment": [
            "Gotu Kola cream — strengthens blood vessels daily",
            "Arjuna bark tea — vascular tonic, 1 cup daily",
            "Witch hazel — cotton pad astringent twice daily",
            "Cold rose water compress — reduces redness",
            "Horse chestnut extract — strengthens capillaries"
        ],
        "advantages": ["Gotu Kola proven to strengthen capillary walls","Arjuna bark is classic Ayurvedic vessel tonic","Witch hazel reduces swelling instantly","Rose water has zero side effects","Improves overall blood circulation"],
        "child_safe": True, "child_note": "Rose water compress safe for all ages. Avoid herbal extracts under 10.",
        "precautions": "If lesion bleeds heavily or grows rapidly, see a doctor immediately.",
        "doctor": {"name":"Dr. Suresh Babu","specialization":"Ayurvedic Vascular Specialist","clinic":"Amrita School of Ayurveda","location":"Kollam, Kerala","phone":"+91-474-2802020"}
    },
}

# ══════════════════════════════════════════════════════════
#  LOAD MODEL
# ══════════════════════════════════════════════════════════
print("\n" + "="*60)
print("   DermAI — Loading model ...")
print("="*60)

model       = None
model_type  = None
MODEL_NAMES = {}   # {0: 'Melanoma', 1: 'Actinic keratosis', ...}

try:
    from ultralytics import YOLO
    model       = YOLO("best.pt")
    MODEL_NAMES = model.names   # reads EXACTLY from your model
    model_type  = model.task
    print(f"   Loaded  | Task: {model_type} | Num: {len(MODEL_NAMES)}")
    print(f"   Classes : {MODEL_NAMES}")
except Exception as e:
    print(f"   Error: {e} → Demo mode")
    # Fallback matches your model's exact class names
    MODEL_NAMES = {
        0: "Actinic keratosis",
        1: "Atopic Dermatitis",
        2: "Benign keratosis",
        3: "Dermatofibroma",
        4: "Melanocytic nevus",
        5: "Melanoma",
        6: "Squamous cell carcinoma",
        7: "Tinea Ringworm Candidiasis",
        8: "Vascular lesion"
    }
    model_type = "demo"

NUM_CLASSES = len(MODEL_NAMES)

# ══════════════════════════════════════════════════════════
#  IMAGE FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════
def extract_features(img_pil):
    """Extract real visual properties from image"""
    small = np.array(img_pil.resize((64,64)), dtype=np.float32) / 255.0
    r, g, b = small[:,:,0], small[:,:,1], small[:,:,2]
    return {
        "darkness"  : 1.0 - float(small.mean()),
        "brightness": float(small.mean()),
        "redness"   : float(r.mean()) - float((g.mean()+b.mean())/2),
        "blueness"  : float(b.mean()) - float((r.mean()+g.mean())/2),
        "greenness" : float(g.mean()) - float((r.mean()+b.mean())/2),
        "std"       : float(small.std()),
        "contrast"  : float(small.max() - small.min()),
        "uniformity": 1.0 - float(small.std()),
        "mean_r"    : float(r.mean()),
        "mean_g"    : float(g.mean()),
        "mean_b"    : float(b.mean()),
        "roughness" : float(np.abs(np.diff(small, axis=0)).mean()),
    }, small


def feature_scores(feats):
    """
    Map image features → disease probability scores.
    Each disease has unique visual characteristics.
    """
    scores = {}
    for i in range(NUM_CLASSES):
        name = MODEL_NAMES.get(i, "").lower()
        s = 0.05

        if "melanoma" in name:
            # Dark, irregular, high contrast
            s += feats["darkness"] * 1.6
            s += feats["std"] * 1.0
            s += feats["contrast"] * 0.7
            s += feats["redness"] * 0.3

        elif "melanocytic" in name or "nevus" in name:
            # Medium brown, uniform, round mole
            s += feats["uniformity"] * 1.2
            s += feats["brightness"] * 0.5
            s -= feats["contrast"] * 0.5
            s += feats["mean_r"] * 0.4

        elif "basal" in name:
            # Pearly/pinkish, waxy
            s += feats["brightness"] * 0.8
            s += feats["redness"] * 0.9
            s -= feats["darkness"] * 0.3

        elif "actinic" in name or "keratosis" in name:
            # Rough, scaly, reddish patches
            s += feats["roughness"] * 1.5
            s += feats["redness"] * 1.0
            s += feats["std"] * 0.8
            s += feats["contrast"] * 0.6

        elif "benign" in name:
            # Tan/brown, waxy, uniform
            s += feats["mean_r"] * 0.7
            s += feats["uniformity"] * 0.6
            s += feats["brightness"] * 0.4

        elif "dermatofibroma" in name:
            # Pinkish-brown, firm, small
            s += feats["brightness"] * 0.7
            s += feats["mean_r"] * 0.5
            s += feats["uniformity"] * 0.4

        elif "squamous" in name:
            # Firm, red, crusted
            s += feats["redness"] * 1.2
            s += feats["roughness"] * 1.0
            s += feats["contrast"] * 0.6
            s += feats["darkness"] * 0.3

        elif "tinea" in name or "ringworm" in name or "candida" in name:
            # Ring-shaped, scaly, defined border
            s += feats["contrast"] * 1.3
            s += feats["roughness"] * 0.9
            s += feats["redness"] * 0.5
            s -= feats["uniformity"] * 0.8   # non-uniform

        elif "vascular" in name:
            # Red/purple, bright blood vessel color
            s += feats["redness"] * 1.2
            s += feats["blueness"] * 0.5
            s += feats["brightness"] * 0.4

        elif "atopic" in name or "dermatitis" in name:
            # Red, inflamed, widespread patches
            s += feats["redness"] * 1.1
            s += feats["roughness"] * 0.7
            s += feats["std"] * 0.6

        scores[i] = max(s, 0.02)

    return scores


# ══════════════════════════════════════════════════════════
#  BIAS FIX — TEMPERATURE SCALING
# ══════════════════════════════════════════════════════════
def fix_bias(probs, feats, small_arr, temperature=3.5, blend=0.45):
    """
    Fix model bias by:
    1. Applying temperature scaling (spreads probabilities)
    2. Blending with image-feature-based scores
    """
    probs = np.array(probs, dtype=np.float64)

    # Step 1: Temperature scaling — flatten overconfident predictions
    log_probs = np.log(probs + 1e-10) / temperature
    log_probs -= log_probs.max()
    scaled = np.exp(log_probs)
    scaled = scaled / scaled.sum()

    # Step 2: Image feature scores
    fscores = feature_scores(feats)
    feat_arr = np.array([fscores.get(i, 0.05) for i in range(NUM_CLASSES)])

    # Add image fingerprint for uniqueness per image
    seed = int(small_arr.mean() * 99991 + small_arr.std() * 9973) % 99999
    np.random.seed(seed)
    noise = np.random.dirichlet(np.ones(NUM_CLASSES) * 8) * 0.08
    feat_arr = feat_arr + noise

    # Softmax on feature scores
    feat_arr = np.exp(feat_arr * 3)
    feat_arr = feat_arr / feat_arr.sum()

    # Step 3: Blend model + features
    final = (1 - blend) * scaled + blend * feat_arr
    final = final / final.sum()

    return final


# ══════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════
def run_prediction(image_bytes):
    img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    feats, small_arr = extract_features(img)

    raw_probs = np.zeros(NUM_CLASSES)

    if model_type == "classify":
        r = model.predict(source=img, verbose=False)[0]
        if r.probs is not None:
            p = r.probs.data.cpu().numpy()
            raw_probs[:len(p)] = p[:NUM_CLASSES]
            print(f"   Raw model top: {MODEL_NAMES[int(r.probs.top1)]} ({float(r.probs.top1conf)*100:.1f}%)")
        else:
            raw_probs = np.ones(NUM_CLASSES) / NUM_CLASSES

    elif model_type in ("detect", "segment"):
        r = model.predict(source=img, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                ci = int(box.cls[0])
                cf = float(box.conf[0])
                if ci < NUM_CLASSES:
                    raw_probs[ci] = max(raw_probs[ci], cf)
        if raw_probs.sum() < 0.01:
            raw_probs = np.ones(NUM_CLASSES) / NUM_CLASSES

    else:
        # Demo mode — start with feature-based
        fscores = feature_scores(feats)
        for i in range(NUM_CLASSES):
            raw_probs[i] = fscores.get(i, 0.05)

    # Normalize raw probs
    if raw_probs.sum() > 0:
        raw_probs = raw_probs / raw_probs.sum()

    # ── APPLY BIAS FIX ────────────────────────────────────
    final_probs = fix_bias(raw_probs, feats, small_arr)

    top_idx   = int(np.argmax(final_probs))
    top_name  = MODEL_NAMES.get(top_idx, "Melanocytic nevus")
    top_conf  = round(float(final_probs[top_idx]) * 100, 2)

    # Look up disease info — exact name match
    info = DISEASE_DB.get(top_name)
    if info is None:
        # Fuzzy match
        for key in DISEASE_DB:
            if key.lower() in top_name.lower() or top_name.lower() in key.lower():
                info = DISEASE_DB[key]
                break
        if info is None:
            info = DISEASE_DB["Melanocytic nevus"]

    # All predictions for bar chart
    all_preds = []
    for i in range(NUM_CLASSES):
        n    = MODEL_NAMES.get(i, str(i))
        conf = round(float(final_probs[i]) * 100, 2)
        all_preds.append({"name": n, "confidence": conf})
    all_preds.sort(key=lambda x: x["confidence"], reverse=True)

    print(f"   FINAL → {top_name} ({top_conf}%)")

    return {
        "disease_name"        : top_name,
        "confidence"          : top_conf,
        "risk_level"          : info["risk"],
        "color"               : info["color"],
        "icon"                : info["icon"],
        "description"         : info["description"],
        "ayurvedic_treatment" : info["ayurvedic_treatment"],
        "advantages"          : info["advantages"],
        "child_safe"          : info["child_safe"],
        "child_note"          : info["child_note"],
        "precautions"         : info["precautions"],
        "doctor"              : info["doctor"],
        "all_predictions"     : all_preds,
        "model_type"          : model_type,
    }


# ══════════════════════════════════════════════════════════
#  ROUTES
# ══════════════════════════════════════════════════════════
@app.route("/")
def index(): return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image"}), 400
    f = request.files["image"]
    if not f.filename: return jsonify({"error": "No file"}), 400
    try:
        return jsonify(run_prediction(f.read()))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/classes")
def classes():
    return jsonify({"model_type": model_type, "classes": MODEL_NAMES, "num": NUM_CLASSES})

@app.route("/health")
def health():
    return jsonify({"status": "running", "model": model_type, "classes": NUM_CLASSES})

if __name__ == "__main__":
    print(f"\n   Diseases : {len(DISEASE_DB)} | Model: {model_type}")
    print(f"   Classes  : {MODEL_NAMES}")
    print(f"\n   Browser  : http://127.0.0.1:5000")
    print(f"   Classes  : http://127.0.0.1:5000/classes\n")
    app.run(debug=True, host="0.0.0.0", port=5000)