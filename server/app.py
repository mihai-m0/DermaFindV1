import tensorflow as tf
import numpy as np
import io
import os
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
# Importăm funcția de preprocesare specifică.
# Dacă modelul tău este MobileNet, schimbă linia de mai jos la:
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications.efficientnet import preprocess_input

# =======================================================
# CONFIGURARE ȘI DATE GLOBALE
# =======================================================

# 1. Calea către fișierul modelului tău .h5
MODEL_PATH = 'skin_cancer_B3_BALANCED.h5'
# 2. Dimensiunea exactă de intrare a modelului
IMAGE_SIZE = (300, 300)
# 3. Lista numelor claselor, ÎN ORDINEA CORECTĂ A MODELULUI TĂU
CLASS_NAMES = ['bcc', 'bkl', 'mel', 'nv']

# Maparea numelor scurte la numele complete pentru afișarea în Frontend
FULL_NAMES = {
    'bcc': 'Carcinom Bazocelular (BCC)',
    'bkl': 'Keratoză Benignă (BKL)',
    'mel': 'Melanom (MEL)',
    'nv': 'Nev Melanocitar (NV)'
}

# =======================================================
# ÎNCĂRCAREA MODELULUI (O SINGURĂ DATĂ)
# =======================================================
ai_model = None
try:
    if os.path.exists(MODEL_PATH):
        # Încărcarea modelului Keras/TensorFlow
        # Folosim compile=False pentru a accelera încărcarea dacă nu facem fine-tuning
        ai_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("--- Server AI GATA ---")
        print(f"Modelul '{MODEL_PATH}' a fost încărcat cu succes. Număr de straturi: {len(ai_model.layers)}")
    else:
        print(f"--- EROARE CRITICĂ --- Fisierul model nu a fost găsit la: {MODEL_PATH}")
        print("Asigură-te că fisierul .h5 este în același director cu app.py.")
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    ai_model = None

# =======================================================
# CONFIGURAREA FLASK ȘI CORS
# =======================================================
app = Flask(__name__)
CORS(app)


# =======================================================
# RUTĂ: PREDICTIA AI
# =======================================================
@app.route('/api/predict', methods=['POST'])
def predict_skin_condition():
    if ai_model is None:
        return jsonify({"success": False, "message": "Eroare: Modelul AI nu s-a putut încărca."}), 500

    if 'file' not in request.files:
        return jsonify({"success": False, "message": "Niciun fișier imagine trimis."}), 400

    try:
        file = request.files['file']
        image_bytes = file.read()
        symptoms = request.form.get('symptoms', 'Nu s-au specificat')

        # --- PREPROCESAREA ---
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMAGE_SIZE)
        image_array = np.array(image)
        input_data = np.expand_dims(image_array.astype('float32'), axis=0)
        input_data = preprocess_input(input_data)

        # --- PREDICȚIA ---
        predictions = ai_model.predict(input_data, verbose=0)
        probs_list = predictions[0].tolist()

        # --- LOGICA PENTRU TOP REZULTATE ---
        # Obținem indicii sortați descrescător după probabilitate
        sorted_indices = np.argsort(probs_list)[::-1]
        
        # Creăm o listă cu toate rezultatele ordonate
        top_results = []
        for i in sorted_indices:
            top_results.append({
                "code": CLASS_NAMES[i],      # ex: 'mel', 'nv'
                "name": FULL_NAMES[CLASS_NAMES[i]], # ex: 'Melanom (MEL)'
                "probability": probs_list[i] # ex: 0.85
            })

        # Selectăm top 1 pentru referință rapidă
        top_1 = top_results[0]

        return jsonify({
            "success": True,
            "top_results": top_results, # Trimitem lista completă ordonată
            "condition": top_1['name'],
            "probability_raw": top_1['probability'],
            "message": "Analiză completă."
        })

    except Exception as err:
        print(f"Eroare la rularea predicției: {err}")
        return jsonify({"success": False, "message": f"Eroare internă: {str(err)}"}), 500

if __name__ == '__main__':
    # Rulați serverul doar dacă modelul s-a încărcat cu succes
    if ai_model is not None:
        # Puteți schimba portul și host-ul aici
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("Serverul nu a putut porni deoarece modelul nu s-a încărcat.")