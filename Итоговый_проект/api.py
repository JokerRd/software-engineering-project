from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from starlette import status
from starlette.responses import JSONResponse

from model import get_stat_by_sentences

from model import calculate_average_score

app = FastAPI()


class CompareTextRequest(BaseModel):
    first_text: str = Field(min_length=1)
    second_text: str = Field(min_length=1)
    separators: list[str] = Field(min_items=1, default=['.', '!', '?'])
    is_percent: bool = Field(default=False)


class ShortResult(BaseModel):
    similarity: float | str


@app.post("/model/text/analysis/short", summary="Метод расчитывает сходство двух текстов")
def calculate_text_similarity(compare_text: CompareTextRequest):
    similarity_tensor = calculate_average_score(compare_text.first_text,
                                                compare_text.second_text,
                                                compare_text.separators)
    similarity = round(float(similarity_tensor), 6)
    if compare_text.is_percent:
        return ShortResult(similarity=convert_to_percent(similarity))
    return ShortResult(similarity=similarity)


@app.post("/model/text/analysis/full", summary="Метод расчитывает сходство предложений двух "
                                      "текстов по принципу каждое с каждым")
def get_full_stat_compare_text(compare_text: CompareTextRequest):
    stats = get_stat_by_sentences(compare_text.first_text, compare_text.second_text, compare_text.separators)
    if compare_text.is_percent:
        return list(map(lambda item: modify_stat(item), stats))
    return stats


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    custom = list(map(lambda item: {"field": item['loc'][-1], "message": item['msg']}, exc.errors()))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=jsonable_encoder({"validation_errors": custom, "body": exc.body}),
    )


def modify_stat(stat):
    stat['similarity'] = convert_to_percent(stat['similarity'])
    return stat


def convert_to_percent(similarity: float):
    rounded_percent_similarity = round(similarity * 100, 2)
    return f"{rounded_percent_similarity}%"
