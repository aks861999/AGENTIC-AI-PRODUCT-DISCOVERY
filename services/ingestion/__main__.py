"""
Allow running the ingestion pipeline as a module:
    python -m ingestion
    python -m ingestion.pipeline   (also works via pipeline.py __main__ guard)
"""
from ingestion.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
