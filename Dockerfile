# Az alap kép, ami CUDA-t és Python-t is tartalmaz
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Frissítsük a Python-t a legújabb verzióra
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-distutils bash \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --config python3 \
    && apt-get clean

ENV PYTHONPATH=/app

# Adjunk hozzá egy környezeti változót a lejárati dátumhoz
ENV EXIT_AFTER_DATE=2025-10-26

# Telepítse a függőségeket a requirements.txt fájlból
COPY ./requirements.dock /app/
WORKDIR /app
RUN pip install --default-timeout=100 -r requirements.dock
RUN python3 -m nltk.downloader punkt

# Másolja a fordító scriptet a konténerbe
COPY compile_all.sh /app/
# Másolja az alkalmazás kódját a konténerbe
COPY ./ollama_streaming.py /app/
COPY ./neo4jrag.py /app/
COPY ./model_manager.py /app/
COPY ./content_manager.py /app/
COPY ./static /app/static
COPY ./templates /app/templates
COPY ./.env /app/
# Futtasd a kód fordítását és az eredeti fájlok eltávolítását
RUN bash /app/compile_all.sh 
RUN ls -lai /app && ls -lai /app/__pycache__
# Az alkalmazás portjának nyitása
EXPOSE 8000

# Hozzunk létre egy Python scriptet futtatásra
COPY start.py /app/start.py

# Az alkalmazás indítása
CMD ["python3", "/app/start.py"]

