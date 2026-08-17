import os
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from semantic_match import build_input, similarity_main, ModelServerError
from graph_grouping import cluster_main


app = FastAPI()


class Item(BaseModel):
    id: str
    text: str
    total_count: int = Field(default=0 ,ge=0)


class Request(BaseModel):
    model_server_url: str = os.environ.get("MODEL_SERVER_URL", "http://localhost:11434/v1/embeddings")
    model_name: str = os.environ.get("MODEL_NAME", "bge-m3")
    threshold: float = Field(ge=0, le=1)
    top_k: int = Field(gt=0)
    items: list[Item]


@app.exception_handler(RequestValidationError)
def validation_error_handler(request, exception):
    return JSONResponse({"detail":exception.errors()}, status_code=400)


@app.post("/similarity")
def similarity(request: Request):
    try:
        data = build_input(request.model_dump())
        result = similarity_main(data)

        return result
    except ModelServerError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/cluster")
def cluster(request: Request):
    try:
        data = build_input(request.model_dump())
        result = cluster_main(data)
        
        return result
    except ModelServerError as e:
        raise HTTPException(status_code=502, detail=str(e))