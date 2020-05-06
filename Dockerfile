FROM ufoym/deepo:pytorch-py36-cu100

RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh

RUN chmod 770 ./Anaconda3-5.3.1-Linux-x86_64.sh

ARG DEBIAN_FRONTEND=noninteractive

RUN yes yes | ./Anaconda3-5.3.1-Linux-x86_64.sh

ENV PATH /yes/bin/:$PATH

RUN conda update conda

COPY environment.yml /tmp/

RUN conda env create -f /tmp/environment.yml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
