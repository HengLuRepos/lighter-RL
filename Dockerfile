FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update
RUN apt-get install -y libglfw3
RUN git clone https://github.com/HengLuRepos/lighter-RL.git
WORKDIR /lighter-RL
RUN pip install -r requirements.txt
