# DeepBurst Sub Task 004 Pipeline 备份

本目录包含执行 `run_sub_task_004_pipeline.sh` 所需的所有代码和脚本文件。

## 文件列表

### 核心执行脚本
- `run_sub_task_004_pipeline.sh` - 主流程脚本，一键执行完整流程

### 步骤脚本
- `test_torch_fake_quant.py` - 步骤0: 生成 fake quant scale (best_scale_dict.json)
- `quantize_mixed_precision.py` - 步骤1: 混合精度量化
- `remove_excluded_layers_qdq.py` - 步骤1.5: 移除被排除层的 Q/DQ 节点
- `sub_task_002_modify_qdq_v2.py` - 步骤2: 修改ONNX模型
- `sub_task_002_inference.py` - 步骤3: 推理与评估

### 依赖模块
- `model_DeepBurst_v7.py` - DeepBurst 模型定义
- `buildingblocks_v6.py` - 编码器/解码器模块
- `block_FSCB.py` - FSCB 注意力模块
- `block_BAM.py` - BAM 注意力模块
- `data_process.py` - 数据加载和预处理
- `utils.py` - 工具函数
- `ssim_multi_threads.py` - SSIM 计算
- `trt_utils.py` - TensorRT 工具函数
- `build_trt_engine_v2.py` - TensorRT 引擎构建

## 使用方法

### 1. 设置目录结构

在 backup 目录下创建必要的软链接和目录:

```bash
cd /gammadisk/liuyijiang/research/deepburst/DeepBurst_claude_extract/torch_example_backup

# 创建输出目录
mkdir -p int8_output
mkdir -p results

# 创建软链接 (指向原始数据)
ln -s ../torch_example/models ./models
ln -s ../torch_example/datasets ./datasets
ln -s ../torch_example/GT ./GT
```

### 2. 运行脚本

```bash
# 加载环境
source /gammadisk/liuyijiang/research/deepburst/DeepBurst_claude_extract/envs.sh

# 执行脚本
./run_sub_task_004_pipeline.sh encoder1,decoder2,decoder3 zoom3
```

## 执行参数说明

```bash
# 不排除任何层 (zoom1模型)
./run_sub_task_004_pipeline.sh

# 排除指定层 (zoom1模型)
./run_sub_task_004_pipeline.sh encoder2

# 排除指定层并指定模型
./run_sub_task_004_pipeline.sh encoder1,decoder2,decoder3 zoom3
```

参数说明:
- 第一个参数: 要排除的层名，逗号分隔 (如 `encoder1,decoder2,decoder3`)
- 第二个参数: 模型名称 (`zoom1` 或 `zoom3`)

## 流程说明

1. **步骤0**: 生成 fake quant scale (best_scale_dict.json)
2. **步骤1**: 混合精度量化
3. **步骤1.5**: 移除被排除层的 Q/DQ 节点
4. **步骤2**: 修改ONNX模型 (使用生成的 scale)
5. **步骤3**: 推理与评估

## 依赖分析

详细依赖分析请查看: `DEPENDENCY_ANALYSIS.md`

### Python 依赖 (通过 envs.sh 加载)
- torch
- numpy
- tensorrt
- onnx
- modelopt
- tifffile
- tqdm
- scikit-image (skimage)
- pyyaml

### 运行时目录
- `models/` - 软链接至 `../torch_example/models`
- `datasets/` - 软链接至 `../torch_example/datasets`
- `GT/` - 软链接至 `../torch_example/GT`
- `int8_output/` - 输出ONNX和TensorRT引擎
- `results/` - 推理结果输出
- `best_scale_dict.json` - 运行时生成 (fake quant scale)

## 注意事项

- 所有Python代码文件都已包含在备份目录中
- 模型权重、数据集、GT需要通过软链接指向原始目录
- 运行时生成的文件会保存在 backup 目录内
