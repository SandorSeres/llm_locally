# Az alap kép, ami CUDA-t és Python-t is tartalmaz
FROM nvidia/cuda:11.0.3-base-ubuntu20.04

# Telepítse a Python-t és a pip-et
RUN apt-get update && apt-get install -y python3 python3-pip bash
ENV PYTHONPATH=/app

# Telepítse a függőségeket a requirements.txt fájlból
COPY ./requirements.dock /app/
WORKDIR /app
RUN pip3 install --default-timeout=100 -r requirements.dock
RUN python3 -m nltk.downloader punkt

# Másolja a fordító scriptet a konténerbe
COPY compile_all.sh /app/
# Másolja az alkalmazás kódját a konténerbe
COPY ./ollama_streaming.py /app/
COPY ./neo4jrag.py /app/
COPY ./model_manager.py /app/
COPY ./static /app/static
COPY ./templates /app/templates
COPY ./.env /app/
# Futtasd a kód fordítását és az eredeti fájlok eltávolítását
RUN bash /app/compile_all.sh 
RUN ls -lai /app && ls -lai /app/__pycache__
# Az alkalmazás portjának nyitása
EXPOSE 8000

# Indítsa el az alkalmazást
CMD ["python3", "/app/__pycache__/ollama_streaming.cpython-38.pyc"]
