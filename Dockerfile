# Az alap kép, ami CUDA-t és Python-t is tartalmaz
FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
CMD nvidia-smi

# Telepítse a Python-t és a pip-et
RUN apt-get update && apt-get install -y python3 python3-pip

# Másolja a függőségkezelő fájlt a konténerbe
COPY ./requirements.dock /app/

# Állítsa be a munkakönyvtárat
WORKDIR /app

# Telepítse a függőségeket a requirements.txt fájlból
RUN pip3 install --default-timeout=100  -r requirements.dock
RUN python3 -m nltk.downloader punkt
# Másolja az alkalmazás kódját a konténerbe
COPY ./streaming.py /app/
COPY ./neo4jrag.py /app/
COPY ./llama_streaming.py /app/
COPY ./static /app/static
COPY ./.env /app/
# Az alkalmazás portjának nyitása
EXPOSE 8000

# Indítsa el az alkalmazást
CMD ["python3", "/app/streaming.py"]
