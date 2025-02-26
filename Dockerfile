# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG PYT_VER=25.02
FROM nvcr.io/nvidia/pytorch:$PYT_VER-py3 as builder

# Update pip and setuptools
RUN pip install --upgrade pip setuptools  

# Setup git lfs, graphviz gl1(vtk dep)
RUN apt-get update && \
    apt-get install -y git-lfs graphviz libgl1 && \
    git lfs install && \
    pip install torchviz

SHELL ["/bin/bash", "-c"]

RUN git clone -b gpu-codecs https://github.com/akshaysubr/zarr-python.git /opt/zarr-python && \
    cd /opt/zarr-python && \
    pip install .[gpu]

ENV _CUDA_COMPAT_TIMEOUT=90
