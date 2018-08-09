FROM tensorflow/tensorflow:1.7.0-gpu

RUN apt update && apt install -y libopencv-dev python-opencv

RUN pip install pip==9.0.0 && \
    pip install scipy numpy opendr matplotlib scikit-image deepdish>=0.3 opencv-python absl-py ipdb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*