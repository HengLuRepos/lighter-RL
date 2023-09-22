FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN apt-get update
RUN apt-get install -y libglfw3
WORKDIR /workspace
RUN pip install gymnasium==0.29.0  "numpy<1.25,>=1.21" scipy==1.11.1  gymnasium[mujoco] matplotlib==3.7.2  torch_tb_profiler
