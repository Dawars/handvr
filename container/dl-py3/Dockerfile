FROM tensorflow/tensorflow:1.10.0-gpu-py3

RUN apt update && apt install -y libopencv-dev python-opencv python3-tk mesa-common-dev

RUN pip3 install --upgrade pip && \
    pip3 install scipy numpy matplotlib scikit-image opencv-python lucid>=0.2.3 \
    http://download.pytorch.org/whl/cu92/torch-0.4.1-cp35-cp35m-linux_x86_64.whl torchvision \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /projects
#ENV PYTHONPATH=$PYTHONPATH:/projects:/projects/handvr
ENV DISPLAY :0

