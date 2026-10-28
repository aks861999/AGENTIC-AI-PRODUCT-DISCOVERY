"""
Allow running the ingestion pipeline as a module:
    python -m ingestion [--index INDEX]
    python -m ingestion.pipeline [--index INDEX]   (also works via pipeline.py __main__ guard)
"""
from ingestion.pipeline import main

if __name__ == "__main__":
    main()
