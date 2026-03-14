# 依赖分析报告

## 脚本执行流程分析

执行命令: `./run_sub_task_004_pipeline.sh encoder1,decoder2,decoder3 zoom3`

---

### 步骤0: test_torch_fake_quant.py (生成 fake quant scale)

**需要的主代码文件 (本地模块):**
- ✅ test_torch_fake_quant.py (已有)
- ✅ model_DeepBurst_v7.py (已有)
- ✅ buildingblocks_v6.py (已有)
- ✅ block_FSCB.py (已有)
- ✅ block_BAM.py (已有)
- ✅ utils.py (已有)
- ✅ data_process.py (已有)
- ✅ ssim_multi_threads.py (已有)

**需要的运行时目录/文件:**
- ❌ `models/zoom3/` - 模型权重目录 (需要从原目录链接)
- ❌ `datasets/zoom3/zoom3_P1/` - 数据集 (需要从原目录链接)
- ❌ `GT/` - Ground Truth (用于SSIM计算，需要从原目录链接或创建软链接)
- ✅ `best_scale_dict.json` - 运行时生成

---

### 步骤1: quantize_mixed_precision.py (混合精度量化)

**需要的主代码文件 (本地模块):**
- ✅ quantize_mixed_precision.py (已有)
- ✅ model_DeepBurst_v7.py (已有)
- ✅ data_process.py (已有)

**需要的外部依赖:**
- modelopt.torch.quantization (Python包)
- modelopt.torch._deploy.utils (Python包)

**需要的运行时目录/文件:**
- ❌ `models/zoom3/` - 模型权重
- ❌ `datasets/zoom3/zoom3_P1/` - 数据集
- ✅ `int8_output/` - 输出目录 (需要创建)

---

### 步骤1.5: remove_excluded_layers_qdq.py (移除被排除层的 Q/DQ 节点)

**需要的主代码文件:**
- ✅ remove_excluded_layers_qdq.py (已有)

**需要的运行时文件:**
- ✅ `./int8_output/deepburst_mixed_zoom3_encoder1_decoder2_decoder3.onnx` (由步骤1生成)

---

### 步骤2: sub_task_002_modify_qdq_v2.py (修改ONNX)

**需要的主代码文件:**
- ✅ sub_task_002_modify_qdq_v2.py (已有)
- ✅ trt_utils.py (已有)
- ✅ build_trt_engine_v2.py (已有)

**需要的运行时文件:**
- ✅ `./int8_output/deepburst_mixed_zoom3_encoder1_decoder2_decoder3.onnx` (已修改)
- ✅ `best_scale_dict.json` (由步骤0生成)

---

### 步骤3: sub_task_002_inference.py (推理与评估)

**需要的主代码文件:**
- ✅ sub_task_002_inference.py (已有)
- ✅ data_process.py (已有)
- ✅ utils.py (已有)
- ✅ ssim_multi_threads.py (已有)
- ✅ trt_utils.py (已有)

**需要的运行时文件:**
- ❌ `./int8_output/deepburst_mixed_zoom3_encoder1_decoder2_decoder3_modified_from_best_scale_dict.plan` - TensorRT引擎
- ❌ `datasets/zoom3/zoom3_P1/` - 数据集
- ❌ `GT/` - Ground Truth (用于SSIM)

---

## 总结: 缺少的依赖

### 1. 需要创建软链接的目录
```bash
ln -s ../torch_example/models ./models
ln -s ../torch_example/datasets ./datasets
ln -s ../torch_example/GT ./GT
```

### 2. 需要创建的输出目录
```bash
mkdir -p int8_output
mkdir -p results
```

### 3. Python 依赖包 (通过 envs.sh 加载)
- torch
- numpy
- tensorrt
- onnx
- modelopt
- tifffile
- tqdm
- scikit-image (skimage)
- pyyaml

---

## 完整设置命令

在 backup 目录下执行:

```bash
cd /gammadisk/liuyijiang/research/deepburst/DeepBurst_claude_extract/torch_example_backup

# 创建必要的目录
mkdir -p int8_output
mkdir -p results

# 创建软链接
ln -s ../torch_example/models ./models
ln -s ../torch_example/datasets ./datasets
ln -s ../torch_example/GT ./GT

# 加载环境并运行
source /gammadisk/liuyijiang/research/deepburst/DeepBurst_claude_extract/envs.sh
./run_sub_task_004_pipeline.sh encoder1,decoder2,decoder3 zoom3
```
