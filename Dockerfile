FROM tensorflow/tensorflow:2.15.0-gpu-jupyter

RUN apt-get update && apt-get install -y cmake

COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install -r /workspace/requirements.txt
# COPY . /tf/notebooks

WORKDIR /workspace