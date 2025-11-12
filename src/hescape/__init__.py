from importlib.metadata import version

# src/hescape/__init__.py
try:
    from importlib.metadata import version
    __version__ = version("HESCAPE")
except Exception:
    __version__ = "0.0.0"  # fallback se non è installato come pacchetto

