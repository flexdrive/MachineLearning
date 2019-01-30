FROM python:2.7

WORKDIR /skynet

ADD . /skynet

RUN pip install tensorflow
RUN pip install sklearn
RUN pip install scipy
RUN pip install numpy

EXPOSE 80

CMD ["python", "./model/TensorFlowModel.py"]
