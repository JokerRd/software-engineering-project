FROM python:3.11.4

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt
RUN \
    --mount=type=cache,target=/var/cache/apt \
    pip install --no-cache-dir --upgrade -r /code/requirements.txt
ENV PYTHONPATH /code/app
COPY ./model.py /code/app/model.py
COPY ./api.py /code/app/api.py
COPY ./__init__.py /code/app/__init__.py
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80"]