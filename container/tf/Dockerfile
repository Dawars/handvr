FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

RUN apt update && apt upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y clang cmake build-essential libgoogle-glog-dev libopencv-dev \
    libsnappy-dev libhdf5-serial-dev \
    wget libomp-dev libatlas-base-dev libleveldb-dev libprotobuf-dev libbz2-dev libboost-all-dev protobuf-compiler \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
#cmake .. -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CC_COMPILER=clang -DBUILD_CAFFE=false -DCaffe_INCLUDE_DIRS=/opt/caffe/include/ -DCaffe_LIBS=/opt/caffe/build/lib/libcaffe.so

#ARG boost_version
#ARG boost_dir
#ENV boost_version ${boost_version}
#
#RUN wget https://dl.bintray.com/boostorg/release/${boost_version}/source/${boost_dir}.tar.gz \
#    && tar xfz ${boost_dir}.tar.gz \
#    && rm ${boost_dir}.tar.gz \
#    && cd ${boost_dir} \
#    && ./bootstrap.sh \
#    && ./b2 --without-python --prefix=/usr -j 4 link=shared runtime-link=shared install \
#    && cd .. && rm -rf ${boost_dir} && ldconfig
