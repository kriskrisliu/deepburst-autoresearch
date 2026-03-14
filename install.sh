# conda create -n deepburst_py312_trt10 python=3.12 -y
# conda activate deepburst_py312_trt10

# torch 2.10
pip3 install torch torchvision -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

export CXX=g++-11 && export CC=gcc-11

# gpus
export CUDA_VISIBLE_DEVICES=2 # 3090 gpu

export CUDA_HOME=/usr/local/cuda-12.8 && export PATH=$CUDA_HOME/bin:$PATH && export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# proxy
export http_proxy=http://114.212.121.151:7890 && export https_proxy=http://114.212.121.151:7890 && export all_proxy=http://114.212.121.151:7890 && export no_proxy=localhost,127.0.0.1

pip install -r requirements.txt
