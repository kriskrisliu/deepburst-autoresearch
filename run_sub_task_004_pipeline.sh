#!/bin/bash
# ============================================================
# Sub Task 004 Pipeline: 完整流程（含 fake quant scale 生成）
# 一键运行:
#   1. 生成排除层的 fake quant scale (best_scale_dict.json)
#   2. 混合精度量化
#   3. 修改ONNX (使用生成的 scale)
#   4. 构建TensorRT引擎
#   5. 推理评估
# ============================================================

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 加载环境
source envs.sh

# ============================================================
# 参数解析
# ============================================================
USAGE="用法: $0 <exclude_layers> [model]
参数:
  exclude_layers  - 要排除量化的层名，逗号分隔 (默认: 无，即不排除任何层)
  model           - 模型名称: zoom1 或 zoom3 (默认: zoom1)

示例:
  $0                       # zoom1模型，不排除任何层
  $0 encoder2              # zoom1模型，排除 encoder2
  $0 encoder2 zoom3        # zoom3模型，排除 encoder2"

# 获取排除层参数
EXCLUDE_LAYERS="${1:-}"

# 获取模型参数 (默认 zoom1)
MODEL="${2:-zoom1}"

# 验证模型参数
if [ "$MODEL" != "zoom1" ] && [ "$MODEL" != "zoom3" ]; then
    echo "错误: model 参数必须是 zoom1 或 zoom3"
    echo "$USAGE"
    exit 1
fi

# 根据模型设置数据集路径
if [ "$MODEL" == "zoom1" ]; then
    DATASET_FOLDER="zoom1/zoom1_P1"
else
    DATASET_FOLDER="zoom3/zoom3_P1"
fi

# 生成安全的文件名
if [ -n "$EXCLUDE_LAYERS" ]; then
    SAFE_NAME=$(echo "$EXCLUDE_LAYERS" | tr ',' '_')
else
    SAFE_NAME="none"
fi

echo "============================================================"
echo "Sub Task 004 Pipeline: 完整流程（含 fake quant scale 生成）"
echo "模型: $MODEL"
echo "数据集: $DATASET_FOLDER"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "排除层: $EXCLUDE_LAYERS"
    echo "Safe name: $SAFE_NAME"
else
    echo "排除层: 无（不排除任何层）"
fi
echo "============================================================"

# ============================================================
# 步骤0: 生成 fake quant scale (best_scale_dict.json)
# ============================================================
echo ""
echo "============================================================"
echo "步骤0: 生成 fake quant scale (best_scale_dict.json)"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "排除层: $EXCLUDE_LAYERS"
else
    echo "不排除任何层"
fi
echo "============================================================"

# 构建 exclude_layers 参数
if [ -n "$EXCLUDE_LAYERS" ]; then
    # 将逗号分隔转换为空格分隔，并构建参数
    EXCLUDE_ARGS=$(echo "$EXCLUDE_LAYERS" | tr ',' ' ')
    EXCLUDE_PARAM="--exclude_layers $EXCLUDE_ARGS"
else
    EXCLUDE_PARAM=""
fi

CUDA_VISIBLE_DEVICES=2 python test_torch_fake_quant.py \
  --GPU 2 \
  --patch_xy 256 \
  --burst 8 \
  --overlap_factor 0 \
  --pth_path ./models \
  --denoise_model $MODEL \
  --datasets_path ./datasets \
  --datasets_folder $DATASET_FOLDER \
  --batch_size 4 \
  --num_workers 4 \
  --ssim \
  --fake_quant \
  --no_save \
  $EXCLUDE_PARAM

if [ $? -ne 0 ]; then
    echo "步骤0失败!"
    exit 1
fi

# 验证 best_scale_dict.json 是否生成
if [ ! -f "best_scale_dict.json" ]; then
    echo "错误: best_scale_dict.json 未生成!"
    exit 1
fi

echo "best_scale_dict.json 已生成"
ls -lh best_scale_dict.json

# ============================================================
# 步骤1: 混合精度量化
# ============================================================
echo ""
echo "============================================================"
echo "步骤1: 混合精度量化"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "排除: $EXCLUDE_LAYERS"
else
    echo "不排除任何层"
fi
echo "============================================================"

CUDA_VISIBLE_DEVICES=2 python quantize_mixed_precision.py \
  --pth_path ./models \
  --denoise_model $MODEL \
  --datasets_path ./datasets \
  --datasets_folder $DATASET_FOLDER \
  --output_dir ./int8_output \
  --calib_samples 16 \
  --batch_size 4 \
  --exclude_layers "$EXCLUDE_LAYERS"

if [ $? -ne 0 ]; then
    echo "步骤1失败!"
    exit 1
fi

# ============================================================
# 步骤1.5: 移除被排除层的 Q/DQ 节点 (真正排除量化)
# ============================================================
echo ""
echo "============================================================"
echo "步骤1.5: 移除被排除层的 Q/DQ 节点"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "将为以下层移除 Q/DQ 节点: $EXCLUDE_LAYERS"
else
    echo "不排除任何层，跳过此步骤"
fi
echo "============================================================"

if [ -n "$EXCLUDE_LAYERS" ]; then
    ORIGINAL_ONNX_PATH="./int8_output/deepburst_mixed_${MODEL}_${SAFE_NAME}.onnx"
    CLEANED_ONNX_PATH="./int8_output/deepburst_mixed_${MODEL}_${SAFE_NAME}_cleaned.onnx"

    echo "输入ONNX: $ORIGINAL_ONNX_PATH"
    echo "输出ONNX: $CLEANED_ONNX_PATH"

    python remove_excluded_layers_qdq.py \
      --onnx_path "$ORIGINAL_ONNX_PATH" \
      --output_path "$CLEANED_ONNX_PATH" \
      --exclude_layers "$EXCLUDE_LAYERS"

    if [ $? -ne 0 ]; then
        echo "步骤1.5失败!"
        exit 1
    fi

    # 用清理后的 ONNX 替换原文件
    mv "$CLEANED_ONNX_PATH" "$ORIGINAL_ONNX_PATH"
    echo "已清理被排除层的 Q/DQ 节点，并替换原ONNX文件"
fi

# ============================================================
# 步骤2: 修改ONNX模型 (使用生成的 best_scale_dict.json)
# ============================================================
echo ""
echo "============================================================"
echo "步骤2: 修改ONNX模型 (使用 fake quant 生成的 scale)"
echo "============================================================"

if [ -n "$EXCLUDE_LAYERS" ]; then
    ONNX_PATH="./int8_output/deepburst_mixed_${MODEL}_${SAFE_NAME}.onnx"
    echo "注意: 此ONNX文件已清理被排除层的Q/DQ节点 ($EXCLUDE_LAYERS)"
else
    ONNX_PATH="./int8_output/deepburst_int8_${MODEL}.onnx"
fi

echo "输入ONNX: $ONNX_PATH"

# 检查ONNX文件是否存在
if [ ! -f "$ONNX_PATH" ]; then
    echo "错误: ONNX文件不存在: $ONNX_PATH"
    exit 1
fi

# 使用生成的 best_scale_dict.json 修改ONNX
# 注意: 被排除层的参数会在 sub_task_002_modify_qdq_v2.py 中被过滤
CUDA_VISIBLE_DEVICES=0,1,2 python sub_task_002_modify_qdq_v2.py \
  --onnx_path "$ONNX_PATH" \
  --output_dir ./int8_output \
  --json_values best_scale_dict.json \
  --exclude_layers "$EXCLUDE_LAYERS" \
  --denoise_model $MODEL \
  --GPU 2

if [ $? -ne 0 ]; then
    echo "步骤2失败!"
    exit 1
fi

# ============================================================
# 步骤3: 推理与评估
# ============================================================
echo ""
echo "============================================================"
echo "步骤3: 推理与评估 (带SSIM)"
echo "============================================================"

# 确定引擎文件路径 (添加模型名以区分 zoom1/zoom3)
if [ -n "$EXCLUDE_LAYERS" ]; then
    ENGINE_PATH="./int8_output/deepburst_mixed_${MODEL}_${SAFE_NAME}_modified_from_best_scale_dict.plan"
else
    ENGINE_PATH="./int8_output/deepburst_int8_${MODEL}_modified_from_best_scale_dict.plan"
fi

echo "使用引擎: $ENGINE_PATH"

# 检查引擎文件是否存在
if [ ! -f "$ENGINE_PATH" ]; then
    echo "错误: 引擎文件不存在: $ENGINE_PATH"
    echo "尝试查找替代文件..."
    if [ -n "$EXCLUDE_LAYERS" ]; then
        ENGINE_PATH=$(ls ./int8_output/deepburst_mixed_${MODEL}_${SAFE_NAME}_modified*.plan 2>/dev/null | tail -1)
    else
        ENGINE_PATH=$(ls ./int8_output/deepburst_int8_${MODEL}_modified*.plan 2>/dev/null | tail -1)
    fi
    if [ -z "$ENGINE_PATH" ]; then
        echo "错误: 找不到引擎文件"
        exit 1
    fi
    echo "使用找到的引擎文件: $ENGINE_PATH"
fi

CUDA_VISIBLE_DEVICES=0,1,2 python sub_task_002_inference.py \
  --engine_path "$ENGINE_PATH" \
  --datasets_path ./datasets \
  --datasets_folder $DATASET_FOLDER \
  --denoise_model $MODEL \
  --batch_size 4 \
  --num_workers 4 \
  --patch_xy 256 \
  --GPU 2 \
  --ssim

if [ $? -ne 0 ]; then
    echo "步骤3失败!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Sub Task 004 Pipeline 完成!"
echo "============================================================"
if [ -n "$EXCLUDE_LAYERS" ]; then
    echo "排除层: $EXCLUDE_LAYERS"
else
    echo "不排除任何层"
fi
echo "结果文件:"
echo "  - best_scale_dict.json (fake quant 生成)"
echo "  - TensorRT引擎: $ENGINE_PATH"
echo "  - 推理结果: ./results/"
echo ""
echo "对比基线:"
echo "  - PyTorch FP16: SSIM = 0.999998"
echo "============================================================"
