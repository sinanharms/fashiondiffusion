import fastapi
import PIL.Image as Image
import streamlit as st
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from webui.dataexplorer import data_explorer

app = FastAPI(title="Fashion Diffusion", description="")


@app.post("/")
def root():
    return JSONResponse(status_code=200, content={"message": "not implemented"})


class InputPrompt(BaseModel):
    text: str


@app.post("/inference")
def inference(data: InputPrompt):
    raise NotImplementedError


@app.post("/generate")
def generate():
    raise NotImplementedError


@app.get("/dataexplorer", response_class=Response)
async def data_explorer():
    streamlit_output = st._server._get_layout(data_explorer)._repr_html_()
    return Response(content=streamlit_output, media_type="text/html")
