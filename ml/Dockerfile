FROM python:3.13-bullseye

# RUN set -x && \
#     apt-get update && \
#     apt-get install -y software-properties-common && \
#     add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) universe" && \
#     add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse" && \
#     apt-get update && \
#     apt-get install -y gfortran build-essential libopenblas-base libopenblas-dev

# Install dependencies
RUN pip install \
    psycopg2-binary \
    # gensim \
    pandas

