#!/bin/bash
docker build -t local_llm -f Dockerfile  .
docker run -p 8000:8000 --gpus all --rm  local_llm
