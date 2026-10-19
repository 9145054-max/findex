import os
from pathlib import Path

class Settings:
    INDEX_PATH: Path = Path(os.getenv("INDEX_PATH", "index.bin"))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", "data"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()