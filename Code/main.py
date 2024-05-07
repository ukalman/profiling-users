from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from bert_race import text_to_loader, tokenizer, DataLoader, predict_probabilities, return_results, MAX_LEN, model

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001"],  # Adjust the port if your HTML file is served on a different one
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

class Tweet(BaseModel):
    tweet: str


class User(BaseModel):
    age: int
    education: str
    gender: str
    race: str
    sex_or: str


@app.get("/")
def read_root():
    return FileResponse('./static/index.html')


@app.post("/predict")
def predict_user(text: Tweet):
    x = DataLoader(text_to_loader(tokenizer, [text.tweet], MAX_LEN), batch_size=1)
    p = predict_probabilities(model, x)
    result = return_results(text, [""], p)

    return result


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
