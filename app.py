from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from databricks import sql

app = Flask(__name__)
CORS(app)

# ===== CONFIG DATABRICKS =====
server_hostname = "YOUR_HOST"
http_path = "YOUR_HTTP_PATH"
access_token = "YOUR_TOKEN"

# ===== CONNECT =====
def get_conn():
    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=access_token
    )

# ===== AI =====
def get_embedding(image_bytes):
    url = "https://bufalo-api-973102760389.asia-southeast1.run.app/predict"

    files = {
        "file": ("img.jpg", image_bytes, "image/jpeg")
    }

    res = requests.post(url, files=files)

    if res.status_code != 200:
        return None

    return res.json().get("embedding")

# ===== COSINE =====
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))

# ===== REGISTER =====
@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")
    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return "No face", 400

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO face_db.faces VALUES (?, ?)",
        (name, emb)
    )

    cursor.close()
    conn.close()

    return "Saved"

# ===== RECOGNIZE =====
@app.route("/recognize", methods=["POST"])
def recognize():

    emb = request.json.get("embedding")

    conn = get_conn()
    cursor = conn.cursor()

    cursor.execute("SELECT name, embedding FROM face_db.faces")

    best_name = "Unknown"
    best_score = 0

    for row in cursor.fetchall():

        name = row[0]
        emb_db = row[1]

        score = cosine(emb, emb_db)

        if score > best_score:
            best_score = score
            best_name = name

    cursor.close()
    conn.close()

    if best_score > 0.5:
        return jsonify({"name": best_name})

    return jsonify({"name": "Unknown"})
