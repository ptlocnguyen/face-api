from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import os
import json
import datetime
from databricks import sql

app = Flask(__name__)

# QUAN TRỌNG: cho phép web gọi
CORS(app)

# ===== ENV =====
HOST = os.getenv("DATABRICKS_HOST")
PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")
AI_URL = os.getenv("AI_URL")

# ===== DB =====
def get_conn():
    return sql.connect(
        server_hostname=HOST,
        http_path=PATH,
        access_token=TOKEN
    )

# ===== AI =====
def get_embedding(image_bytes):

    files = {
        "file": ("img.jpg", image_bytes, "image/jpeg")
    }

    res = requests.post(AI_URL, files=files)

    if res.status_code != 200:
        return None

    return res.json().get("embedding")

# ===== COSINE =====
def cosine(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    if len(a) != len(b):
        return 0

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")
    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return "No face", 400

    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO face_db.faces VALUES (?, ?)",
        (name, json.dumps(emb))
    )

    cur.close()
    conn.close()

    return "Saved"

# ================= RECOGNIZE (WEB) =================
@app.route("/recognize_image", methods=["POST"])
def recognize_image():

    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"name": "No face"})

    return match_face(emb, save_log=False)

# ================= RECOGNIZE (ESP32) =================
@app.route("/recognize", methods=["POST"])
def recognize():

    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"name": "No face"})

    return match_face(emb, save_log=True)

# ===== MATCH =====
def match_face(emb, save_log):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT name, embedding FROM face_db.faces")

    best_name = "Unknown"
    best_score = 0

    for row in cur.fetchall():

        name = row[0]
        emb_db = json.loads(row[1])

        score = cosine(emb, emb_db)

        if score > best_score:
            best_score = score
            best_name = name

    if best_score > 0.5 and save_log:

        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur.execute(
            "INSERT INTO face_db.logs VALUES (?, ?)",
            (best_name, now)
        )

    cur.close()
    conn.close()

    return jsonify({"name": best_name})

# ================= LOGS =================
@app.route("/logs")
def logs():

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT name, time FROM face_db.logs ORDER BY time DESC LIMIT 50")

    data = []

    for r in cur.fetchall():
        data.append({
            "name": r[0],
            "time": r[1]
        })

    cur.close()
    conn.close()

    return jsonify(data)

# ================= TEST =================
@app.route("/")
def home():
    return "API OK"
