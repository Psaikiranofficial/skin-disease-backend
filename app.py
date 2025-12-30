from fastapi import FastAPI
from routers import skin

app = FastAPI(
    title="Skin Disease Prediction API",
    description="CNN-based Skin Disease Detection",
    version="1.0"
)

app.include_router(skin.router)

@app.get("/")
def root():
    return {"message": "Skin Disease API running"}
