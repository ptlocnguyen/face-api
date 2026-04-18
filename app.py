from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
import os
import json
from databricks import sql
import pytz
import datetime

app = Flask(__name__)

# ===== TIME =====
def get_time_vn():
    tz = pytz.timezone("Asia/Ho_Chi_Minh")
    return datetime.datetime.now(tz)

# ===== CORS =====
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

# ===== ENV =====
HOST = os.getenv("DATABRICKS_HOST")
PATH = os.getenv("DATABRICKS_HTTP_PATH")
TOKEN = os.getenv("DATABRICKS_TOKEN")
AI_URL = os.getenv("AI_URL")

# ===== CACHE (MULTI EMBEDDING) =====
faces_cache = {}

# ===== DB =====
def get_conn():
    return sql.connect(
        server_hostname=HOST,
        http_path=PATH,
        access_token=TOKEN
    )

# ===== LOAD CACHE =====
def load_faces():
    global faces_cache

    print("Loading faces...")

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT name, embedding FROM face_db.faces")

        faces_cache = {}

        for row in cur.fetchall():
            name = row[0]
            emb = json.loads(row[1])

            if name not in faces_cache:
                faces_cache[name] = []

            faces_cache[name].append(emb)

        cur.close()
        conn.close()

        print("Loaded:", len(faces_cache), "users")

    except Exception as e:
        print("LOAD ERROR:", e)

load_faces()

# ===== AI CALL (RETRY) =====
def get_embedding(image_bytes):

    files = {"file": ("img.jpg", image_bytes, "image/jpeg")}

    for _ in range(2):
        try:
            res = requests.post(AI_URL, files=files, timeout=5)

            if res.status_code == 200:
                return res.json().get("embedding")
        except:
            pass

    return None

# ===== COSINE =====
def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ===== MATCH (MULTI EMBEDDING) =====
def match_face(emb):

    best_name = "Unknown"
    best_score = 0

    for name, emb_list in faces_cache.items():

        for emb_db in emb_list:

            score = cosine(emb, emb_db)

            if score > best_score:
                best_score = score
                best_name = name

        if best_score > 0.7:
            break

    print("Best:", best_name, best_score)

    if best_score > 0.6:
        return best_name

    return "Unknown"

# ================= REGISTER =================
@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")
    file = request.files.get("file")

    if not name or not file:
        return "Missing data", 400

    emb = get_embedding(file.read())

    if emb is None:
        return "No face", 400

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "INSERT INTO face_db.faces VALUES (?, ?)",
            (name, json.dumps(emb))
        )

        cur.close()
        conn.close()

        # update cache
        if name not in faces_cache:
            faces_cache[name] = []

        faces_cache[name].append(emb)

        return "Saved"

    except Exception as e:
        print("REGISTER ERROR:", e)
        return "DB Error", 500

# ================= RECOGNIZE =================
@app.route("/recognize_image", methods=["POST"])
def recognize_image():

    file = request.files.get("file")

    emb = get_embedding(file.read())

    if emb is None:
        return jsonify({"name": "No face"})

    name = match_face(emb)

    if name != "Unknown":
        try:
            conn = get_conn()
            cur = conn.cursor()

            now = get_time_vn().strftime("%Y-%m-%d %H:%M:%S")

            cur.execute(
                "INSERT INTO face_db.logs VALUES (?, ?)",
                (name, now)
            )

            cur.close()
            conn.close()
        except:
            pass

    return jsonify({"name": name})

# ================= LOGS =================
@app.route("/logs")
def logs():

    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT name, time FROM face_db.logs LIMIT 20")

        data = [{"name": r[0], "time": r[1]} for r in cur.fetchall()]

        cur.close()
        conn.close()

        return jsonify(data)

    except:
        return jsonify([])

@app.route("/")
def home():
    return "API OK"
