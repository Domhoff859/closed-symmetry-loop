FROM tensorflow/tensorflow:2.7.4-gpu-jupyter

RUN apt-get update && apt-get install -y nano \
            cmake

COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt

WORKDIR /workspace