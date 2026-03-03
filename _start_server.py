"""Start the backend API server."""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import uvicorn
uvicorn.run("app.main:app", host="127.0.0.1", port=8002)
