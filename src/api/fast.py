import fastapi
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(title="Fashion Diffusion", description="")


@app.post("/")
def root():
    return JSONResponse(status_code=200, content={"message": "not implemented"})


@app.post("/inference")
def inference():
    raise NotImplementedError


@app.post("/generate")
def generate():
    raise NotImplementedError
