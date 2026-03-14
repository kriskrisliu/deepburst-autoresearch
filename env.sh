source /gammadisk/liuyijiang_new/miniconda3/etc/profile.d/conda.sh
conda activate deepburst_py312_trt10
export CUDA_VISIBLE_DEVICES=2
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH                          
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# proxy
export http_proxy=http://114.212.121.151:7890
export https_proxy=http://114.212.121.151:7890
export all_proxy=http://114.212.121.151:7890
export no_proxy=localhost,127.0.0.1