import os
import base64
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow requests from your HTML app on any domain

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

PROMPT = """You are reading a JK Paper pallet label (brand: JK Ultima / JK Excel / JK Sparkle etc).
The label prints as 4 identical strips on one sheet — read only ONE strip.

LAYOUT OF EACH STRIP:
  TOP ROW    : COMMODITY-PAPER | GSM | SIZE (CM) | MFG.ON (DD.MM.YYYY) | MRP
  LEFT PANEL : Blue box with JK logo and brand name e.g. Ultima — this is the GRADE
  MIDDLE ROW : QUANTITY (SHEETS) | LOT NO. | MRP
  BOTTOM ROW : NOT FOR RETAIL SALE | NET.WT.(KG) value | PLT.NO.: value | REEL NO.: value

Return ONLY this JSON object, no markdown, no explanation:
{
  "plt":   "PLT NO — bottom-centre, alphanumeric e.g. C526E01332",
  "reel":  "REEL NO — bottom-right, alphanumeric e.g. A26E077003",
  "gsm":   "GSM — top-left, format NNN/GLW e.g. 350/GLW",
  "grade": "Brand name in left blue panel e.g. Ultima",
  "size":  "SIZE in CM e.g. 66.0 X 96.5",
  "wt":    "NET WT KG e.g. 22.30",
  "qty":   "QUANTITY sheets e.g. 0100",
  "lot":   "LOT NO e.g. 2605249/213337",
  "mfg":   "MFG ON date exactly as printed with dots e.g. 28.05.2026",
  "mrp":   "MRP digits only, no rupee symbol, no /CBB e.g. 3568.00"
}

RULES:
- Read PLT NO and REEL NO character by character — never guess.
- PLT NO starts with a letter then digits e.g. C526E01332
- REEL NO starts with a letter then digits e.g. A26E077003
- GSM always ends /GLW e.g. 350/GLW
- MFG date uses dots DD.MM.YYYY — return exactly as printed
- Empty string for any field not clearly visible
- Return ONLY the JSON object"""


@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "service": "BM5 Vision API",
        "version": "1.0",
        "gemini_configured": bool(GEMINI_KEY)
    })


@app.route("/scan", methods=["POST"])
def scan_label():
    # ── Validate request ───────────────────────────────────────────────────
    if not GEMINI_KEY:
        return jsonify({"ok": False, "error": "GEMINI_API_KEY not configured on server"}), 500

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"ok": False, "error": "No image provided — send base64 image in 'image' field"}), 400

    image_b64 = data["image"]

    # Strip data URL prefix if present (data:image/jpeg;base64,...)
    if "," in image_b64:
        image_b64 = image_b64.split(",")[1]

    # ── Call Gemini ────────────────────────────────────────────────────────
    payload = {
        "contents": [{
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": image_b64
                    }
                },
                {
                    "text": PROMPT
                }
            ]
        }],
        "generationConfig": {
            "maxOutputTokens": 512,
            "temperature": 0
        }
    }

    try:
        resp = requests.post(
            f"{GEMINI_URL}?key={GEMINI_KEY}",
            json=payload,
            timeout=30
        )
    except requests.exceptions.Timeout:
        return jsonify({"ok": False, "error": "Gemini request timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"ok": False, "error": f"Network error calling Gemini: {str(e)}"}), 502

    if resp.status_code != 200:
        try:
            err = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            err = resp.text
        return jsonify({"ok": False, "error": f"Gemini error {resp.status_code}: {err}"}), 502

    # ── Parse Gemini response ──────────────────────────────────────────────
    try:
        raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        return jsonify({"ok": False, "error": f"Unexpected Gemini response structure: {str(e)}"}), 502

    # Strip markdown code fences if Gemini added them
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1]
        clean = clean.rsplit("```", 1)[0]
    clean = clean.strip()

    try:
        fields = json.loads(clean)
    except json.JSONDecodeError:
        return jsonify({"ok": False, "error": f"Could not parse Gemini response: {clean[:200]}"}), 502

    # ── Normalise fields ───────────────────────────────────────────────────
    # MFG date: dots → slashes (28.05.2026 → 28/05/2026)
    if fields.get("mfg"):
        fields["mfg"] = fields["mfg"].replace(".", "/")

    # MRP: strip ₹, commas, /CBB
    if fields.get("mrp"):
        fields["mrp"] = (
            fields["mrp"]
            .replace("₹", "")
            .replace(",", "")
            .split("/")[0]
            .strip()
        )

    return jsonify({"ok": True, "fields": fields})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
