# cache.py — no-op stub
async def cache_get(key: str):
    return None

async def cache_set(key: str, value, ttl: int = 300):
    pass

async def ping_redis() -> bool:
    return True

async def close_redis():
    pass