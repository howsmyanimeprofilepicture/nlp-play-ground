FROM python:3.11-slim
WORKDIR /code

RUN pip install --upgrade diffusers accelerate transformers -qqq
COPY ./src ./src

CMD ["python", "./src/main.py"]