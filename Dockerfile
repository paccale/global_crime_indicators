FROM python:3.12-slim
WORKDIR /src

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/serve.py .
COPY src/predict.py .
COPY src/config.py .
COPY src/data_preprocessing.py .

RUN mkdir -p /Data/models
COPY Data/models/RandomForest.pkl /Data/models
COPY Data/models/dict_vectorizer.pkl /Data/models
COPY Data/models/scaler.pkl /Data/models

EXPOSE 5000

CMD ["python", "serve.py"]