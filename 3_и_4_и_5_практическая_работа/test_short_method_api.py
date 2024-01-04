import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

first_raw_text = ("Развитие математики началось вместе с тем, как человек стал использовать абстракции сколько-нибудь "
                  "высокого уровня. Простая абстракция — числа; осмысление того, что два яблока и два апельсина, "
                  "несмотря на все их различия, имеют что-то общее, а именно занимают обе руки одного человека, "
                  "— качественное достижение мышления человека. Кроме того, что древние люди узнали, как считать "
                  "конкретные"
                  "объекты, они также поняли, как вычислять и абстрактные количества, такие, как время: дни, сезоны, "
                  "года. Из элементарного счёта естественным образом начала развиваться арифметика: сложение, "
                  "вычитание,"
                  "умножение и деление чисел.")

second_raw_text = ("Задумка по реализации языка появилась в конце 1980-х годов, а разработка его реализации началась "
                   "в 1989 году сотрудником голландского института CWI Гвидо ван Россумом[39]. Для распределённой "
                   "операционной системы Amoeba требовался расширяемый скриптовый язык, и Гвидо начал разрабатывать "
                   "Python на досуге, позаимствовав некоторые наработки для языка ABC (Гвидо участвовал в разработке "
                   "этого языка, ориентированного на обучение программированию). В феврале 1991 года Гвидо "
                   "опубликовал исходный текст в группе новостей alt.sources[43]. С самого начала Python "
                   "проектировался как объектно-ориентированный язык.")

short_analysis_url = "/model/text/analysis/short"
full_analysis_url = "/model/text/analysis/full"


def test_identical_similarity_when_text_identical():
    response = client.post(short_analysis_url, json={"first_text": first_raw_text, "second_text": first_raw_text})
    assert response.status_code == 200
    assert response.json() == {"similarity": 1}


def test_different_similarity_when_text_different():
    response = client.post(short_analysis_url, json={"first_text": first_raw_text, "second_text": second_raw_text})
    assert response.status_code == 200
    assert response.json() != {"similarity": 1}


@pytest.mark.parametrize("first_text,second_text", [(None, None), ("", ""), (None, ""), ("", None)])
def test_error_when_any_text_none_or_empty(first_text, second_text):
    response = client.post(short_analysis_url, json={"first_text": first_text,
                                                     "second_text": second_text})
    assert response.status_code == 400


def test_identical_similarity_when_text_identical_with_custom_separator():
    text_with_custom_separator = first_raw_text.replace('.', '~')
    response = client.post(short_analysis_url, json={"first_text": text_with_custom_separator,
                                                     "second_text": text_with_custom_separator,
                                                     "separators": ['~']})
    assert response.status_code == 200
    assert response.json() == {"similarity": 1}


def test_error_when_custom_separator_is_empty():
    response = client.post(short_analysis_url, json={"first_text": first_raw_text,
                                                     "second_text": second_raw_text,
                                                     "separators": []})
    assert response.status_code == 400


def test_similarity_with_percent_when_flag_is_percent_true():
    response = client.post(short_analysis_url, json={"first_text": first_raw_text,
                                                     "second_text": second_raw_text,
                                                     "is_percent": True})
    assert response.status_code == 200
    similarity = response.json()["similarity"]
    assert "%" in similarity


def test_similarity_without_percent_when_flag_is_percent_false():
    response = client.post(short_analysis_url, json={"first_text": first_raw_text,
                                                     "second_text": second_raw_text,
                                                     "is_percent": False})
    assert response.status_code == 200
    similarity = str(response.json()["similarity"])
    assert "%" not in similarity
