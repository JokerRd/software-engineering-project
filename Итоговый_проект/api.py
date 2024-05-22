from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette import status
from starlette.responses import JSONResponse
from model import get_stat_by_sentences
from model import calculate_average_score
from utils import convert_to_percent, transform_similarity_to_percent

app = FastAPI()


class CompareTextRequest(BaseModel):
    first_text: str = Field(min_length=1)
    second_text: str = Field(min_length=1)
    separators: list[str] = Field(min_items=1, default=['.', '!', '?'])
    is_percent: bool = Field(default=False)


class ShortResult(BaseModel):
    similarity: float | str


@app.post("/model/text/analysis/short",
          summary="Метод расчитывает сходство двух текстов")
def calculate_text_similarity(compare_text: CompareTextRequest):
    similarity_tensor = calculate_average_score(compare_text.first_text,
                                                compare_text.second_text,
                                                compare_text.separators)
    similarity = round(float(similarity_tensor), 6)
    if compare_text.is_percent:
        return ShortResult(similarity=convert_to_percent(similarity))
    return ShortResult(similarity=similarity)


@app.post("/model/text/analysis/full",
          summary="Метод расчитывает сходство предложений"
                  " двух текстов по принципу каждое с каждым")
def get_full_stat_compare_text(compare_text: CompareTextRequest):
    stats = get_stat_by_sentences(compare_text.first_text,
                                  compare_text.second_text,
                                  compare_text.separators)
    if compare_text.is_percent:
        return list(map(transform_similarity_to_percent, stats))
    return stats


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request,
                                       exc: RequestValidationError):
    custom_error_message = list(map(create_custom_error_message, exc.errors()))
    error_response = create_error_response(custom_error_message, exc.body)
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder(error_response),
    )


def create_custom_error_message(error):
    return {"field": error['loc'][-1], "message": error['msg']}


def create_error_response(error_message, body):
    return {"validation_errors": error_message, "body": body}