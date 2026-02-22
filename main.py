import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI()

# ✅ secure API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -------- Request Model --------
class CommentRequest(BaseModel):
    comment: str

# -------- Response Model --------
class SentimentResponse(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    rating: int

# -------- JSON Schema --------
json_schema = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "neutral"]
        },
        "rating": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["sentiment", "rating"],
    "additionalProperties": False
}

# -------- Endpoint --------
@app.post("/comment", response_model=SentimentResponse)
async def analyze_comment(request: CommentRequest):
    try:
        if not request.comment.strip():
            raise HTTPException(status_code=400, detail="Comment cannot be empty")

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Analyze the sentiment of this comment: {request.comment}",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "sentiment_schema",
                    "schema": json_schema,
                    "strict": True
                }
            }
        )

        return response.output_parsed

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))