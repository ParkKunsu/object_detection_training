# Base image: Ubuntu 20.04 with CUDA 11.8
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04


# Set environment variables for CUDA
ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"


# Default command
CMD ["/bin/bash"]
