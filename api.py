# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from prediction import predictor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow requests from your extension (for dev, allow all origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class URLRequest(BaseModel):
    url: str

@app.post("/predict")
async def predict_url(req: URLRequest):
    result = predictor(req.url)
    label = "good" if result == 0 else "bad"
    return {"label": label}
