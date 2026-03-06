from fastapi import FastAPI
import uvicorn
from backends import auth_backend, data_backend, reports_backend

app = FastAPI()

@app.get("/auth")
def auth_endpoint():
    auth_backend()
    return {"status": "authenticated", "user": "john_doe"}

@app.get("/data")
def data_endpoint():
    data_backend()
    return {"status": "success", "data": [1, 2, 3, 4, 5]}

@app.get("/reports")
def reports_endpoint():
    reports_backend()
    return {"status": "complete", "report": "analytics_data"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)