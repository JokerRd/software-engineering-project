import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

first_raw_text = ("Советская власть стала проводить антирелигиозную политику с начала своего существования[2]. Однако "
                  "руководство большевиков пришло к выводу, что прямое наступление на религию не увенчалось успехом, "
                  "и поэтому было решено подойти к борьбе с ней в более мягких формах. К этому призывали некоторые "
                  "идеологи партии и атеистической пропаганды (Лев Троцкий, Емельян Ярославский, "
                  "Анатолий Луначарский, Иван Скворцов-Степанов)[3]. Частью антирелигиозной кампании стала борьба "
                  "против православных праздников и связанных с ними обычаев, традиционной обрядности за которые всё "
                  "ещё прочно держалось население[4]. В 1922—1923 годах в СССР была проведена централизованная "
                  "кампания по празднованию «комсомольского рождества» («комсвятки», «красное рождество») и "
                  "«комсомольской пасхи», направленных на внедрение «красной обрядности»")

second_raw_text = ("По оценке историка Наталии Лебиной, советское руководство сделало ставку на антирелигиозные "
                   "молодёжные выступления, ставшие своеобразным «заключительным аккордом» атеистической кампании "
                   "1922 года[7]. Для них характерно привлечение прежде всего молодёжи, массовый характер, "
                   "театрализованность[3][8]. Лебина предположила, что такие политизированные действа восходят к "
                   "массовым театрализованным антирелигиозным шествиям, распространённым во времена Великой "
                   "французской буржуазной революции[5]. В октябре 1922 года на Украине и в Белоруссии прошли первые "
                   "антирелигиозные кампании, которые получили освещение в газете «Правда»[9]. Через месяц вышла "
                   "статья Скворцова-Степанова «„Комсомольское рождество“ "
                   "или почему бы нам не справлять религиозные праздники», где была представлена программа проведения "
                   "«Весёлого карнавала с музыкой»[10]")

full_analysis_url = "/model/text/analysis/full"


def test_size_output_is_square_input_length_when_text_identical():
    response = client.post(full_analysis_url, json={"first_text": first_raw_text, "second_text": first_raw_text})
    assert response.status_code == 200
    result = response.json()
    len_text = len(first_raw_text.split('.'))
    assert len(result) == (len_text ** 2)


def test_size_output_equal_multiple_lengths_texts_when_different_text():
    response = client.post(full_analysis_url, json={"first_text": first_raw_text, "second_text": second_raw_text})
    assert response.status_code == 200
    result = response.json()
    len_first_text = len(first_raw_text.split('.'))
    len_second_text = len(second_raw_text.split('.'))
    assert len_first_text * len_second_text == len(result)


@pytest.mark.parametrize("first_text,second_text", [(None, None), ("", ""), (None, ""), ("", None)])
def test_error_when_any_text_none_or_empty(first_text, second_text):
    response = client.post(full_analysis_url, json={"first_text": first_text,
                                                    "second_text": second_text})
    assert response.status_code == 400


def test_size_output_is_square_input_length_with_custom_separator():
    text_with_custom_separator = first_raw_text.replace('.', '~')
    response = client.post(full_analysis_url, json={"first_text": text_with_custom_separator,
                                                    "second_text": text_with_custom_separator,
                                                    "separators": ['~']})
    assert response.status_code == 200
    result = response.json()
    len_text = len(text_with_custom_separator.split('~'))
    assert len(result) == (len_text ** 2)


def test_error_when_custom_separator_is_empty():
    response = client.post(full_analysis_url, json={"first_text": first_raw_text,
                                                    "second_text": second_raw_text,
                                                    "separators": []})
    assert response.status_code == 400


def test_all_similarity_with_percent_when_flag_is_percent_true():
    response = client.post(full_analysis_url, json={"first_text": first_raw_text,
                                                    "second_text": second_raw_text,
                                                    "is_percent": True})
    assert response.status_code == 200
    result = response.json()
    assert all(map(lambda item: "%" in str(item["similarity"]), result))


def test_similarity_without_percent_when_flag_is_percent_false():
    response = client.post(full_analysis_url, json={"first_text": first_raw_text,
                                                    "second_text": second_raw_text,
                                                    "is_percent": False})
    assert response.status_code == 200
    result = response.json()
    assert all(map(lambda item: "%" not in str(item["similarity"]), result))
