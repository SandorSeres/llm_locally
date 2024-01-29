# Az alap kép, ami CUDA-t és Python-t is tartalmaz
FROM nvidia/cuda:11.0.3-base-ubuntu20.04 
CMD nvidia-smi

# Telepítse a Python-t és a pip-et
RUN apt-get update && apt-get install -y python3 python3-pip

# Másolja az alkalmazás kódját és a függőségkezelő fájlt a konténerbe
COPY ./app.py /app/
COPY ./index.html /app/
COPY ./requirements.txt /app/

# Állítsa be a munkakönyvtárat
WORKDIR /app

# Telepítse a függőségeket a requirements.txt fájlból
RUN pip3 install -r requirements.txt

# Az alkalmazás portjának nyitása
EXPOSE 8000

# Indítsa el az alkalmazást
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
