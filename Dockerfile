# Az alap kép, ami CUDA-t és Python-t is tartalmaz
#FROM nvidia/cuda:11.0.3-base-ubuntu20.04
FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV TZ=Europe/Budapest
RUN echo "$TZ" > /etc/timezone && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
# Alap csomagok telepítése
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y python3.11 python3.11-distutils bash \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --config python3 \
    && apt-get clean


# PIP frissítése és alap Python konfiguráció
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --config python3

# CUDA path beállítása (ha szükséges a `vllm` miatt)
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Python csomagok telepítése (először NumPy és Torch, hogy a többi csomag ne hibázzon)
RUN pip3 install --no-cache-dir numpy torch

# Telepítse a függőségeket a requirements.dock fájlból
COPY ./requirements.dock /app/
WORKDIR /app
RUN cat requirements.dock
RUN pip3 install --no-cache-dir -r requirements.dock
RUN python3 -m nltk.downloader punkt

# Másolja a fordító scriptet a konténerbe
COPY compile_all.sh /app/
COPY ./ollama_streaming.py /app/
COPY ./neo4jrag.py /app/
COPY ./model_manager.py /app/
COPY ./content_manager.py /app/
COPY ./static /app/static
COPY ./templates /app/templates
COPY ./.env /app/
COPY ./deploy/run.sh /app/
RUN chmod +x /app/run.sh

# PyArmor telepítése a kód védelméhez
RUN pip install --no-cache-dir pyarmor

# Fordítás és eredeti fájlok eltávolítása
RUN bash -x /app/compile_all.sh && rm /app/compile_all.sh

# Indító script másolása
COPY ./start.py /app/start.py

# Az alkalmazás portjának nyitása
EXPOSE 8000

# Az alkalmazás indítása
CMD ["bash", "/app/run.sh"]

