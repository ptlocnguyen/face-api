import time
from db import load_face_cache
from config import CACHE_REFRESH

face_cache = []
last_load = 0

def refresh_cache():
    global face_cache, last_load
    face_cache = load_face_cache()
    last_load = time.time()
    print("Cache loaded:", len(face_cache))

def get_cache():
    global last_load

    if time.time() - last_load > CACHE_REFRESH:
        refresh_cache()

    return face_cache
