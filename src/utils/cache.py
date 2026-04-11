import json
import os

CACHE_FILE = 'data/news_cache.json'

def get_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)

def get_analysis(title):
    cache = get_cache()
    return cache.get(title)

def set_analysis(title, data):
    cache = get_cache()
    cache[title] = data
    save_cache(cache)
