# app/documents/utils.py
import redis
import time

r = redis.Redis(host="localhost", port=6379, db=0)

def cache_document_result(doc_id: str, result: dict):
    r.set(doc_id, str(result), ex=3600)  # cache for 1 hour

def get_cached_result(doc_id: str):
    data = r.get(doc_id)
    return eval(data) if data else None
