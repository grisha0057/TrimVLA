#!/bin/bash

# 评测 LIBERO Checkpoint
# 使用方法:
#   1. 通过环境变量传递（推荐）:
#      CHECKPOINT_PATH="/path/to/checkpoint" ./eval_libero_latest.sh
#   2. 通过命令行参数传递:
#      ./eval_libero_latest.sh [checkpoint_path]
#   3. 直接运行（使用默认路径）:
#      ./eval_libero_latest.sh

set -e

# 获取 checkpoint 路径（优先级：环境变量 > 命令行参数 > 默认值）
if [ -n "${CHECKPOINT_PATH}" ]; then
    # 使用环境变量中的路径（用户已经设置了）
    :
elif [ $# -ge 1 ]; then
    # 使用命令行参数
    CHECKPOINT_PATH="$1"
else
    # 使用默认checkpoint路径（可以修改为你要评测的checkpoint）
    CHECKPOINT_PATH="/root/workspace/LightVLA/logs/libero_spatial_training/libero_spatial_from1400_20251102_142005/libero_spatial_from1400_20251102_1420052025-11-02 14:20:33.512042--1200_chkpt"
fi

# 验证checkpoint存在
if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "❌ 错误: Checkpoint 不存在: ${CHECKPOINT_PATH}"
    exit 1
fi

echo "============================================"
echo "🎮 评测 LIBERO Checkpoint"
echo "============================================"
echo ""

# 激活 conda 环境
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda activate openvla-oft
echo "✅ 已激活 conda 环境: openvla-oft"

# 渲染配置
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
echo "✅ 使用 OSMesa 软件渲染"

echo "📦 Checkpoint: ${CHECKPOINT_PATH}"
echo "📊 Checkpoint 大小: $(du -sh "${CHECKPOINT_PATH}" | cut -f1)"
echo ""

# 评估配置（这些都有默认值，但建议根据训练配置设置）
EVAL_GPUS=${EVAL_GPUS:-"0,1"}          # 使用GPU（默认：0,1）
NUM_TRIALS=${NUM_TRIALS:-50}             # 每个任务试验次数（默认：20，可设为50获得更稳定结果）
LORA_RANK=${LORA_RANK:-8}               # LoRA rank（默认：8，应与训练配置一致）
PRUNE_MIN_KEEP_RATIO=${PRUNE_MIN_KEEP_RATIO:-0.1}  # 视觉Token筛选比例（留空使用checkpoint配置）

# 从checkpoint路径自动推断输出目录
CHECKPOINT_DIR=$(dirname "${CHECKPOINT_PATH}")
OUTPUT_DIR="${CHECKPOINT_DIR}/eval_logs"
mkdir -p "${OUTPUT_DIR}"

echo "⚙️  评测配置："
echo "  - Checkpoint: ${CHECKPOINT_PATH}"
echo "  - GPU: ${EVAL_GPUS}"
echo "  - 每任务试验次数: ${NUM_TRIALS} (💡 设置为50可获得更稳定结果，但会慢2.5倍)"
echo "  - LoRA Rank: ${LORA_RANK}"
if [ -n "${PRUNE_MIN_KEEP_RATIO}" ]; then
    echo "  - 视觉Token筛选: ${PRUNE_MIN_KEEP_RATIO}"
else
    echo "  - 视觉Token筛选: 使用checkpoint的config.json配置"
fi
echo "  - 日志目录: ${OUTPUT_DIR}"
echo ""

# 设置GPU
export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}

cd /root/workspace/LightVLA

echo "🚀 开始评测..."
echo "============================================"
echo ""

# 记录开始时间
EVAL_START_TIME=$(date +%s)

# 构建评测命令
EVAL_CMD="python -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint \"${CHECKPOINT_PATH}\" \
    --task_suite_name \"libero_spatial\" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank ${LORA_RANK}"

# 如果指定了 prune_min_keep_ratio，添加该参数
if [ -n "${PRUNE_MIN_KEEP_RATIO}" ]; then
    EVAL_CMD="${EVAL_CMD} \
    --prune_min_keep_ratio ${PRUNE_MIN_KEEP_RATIO}"
fi

EVAL_CMD="${EVAL_CMD} \
    --center_crop True \
    --num_trials_per_task ${NUM_TRIALS} \
    --run_id_note \"eval_\$(basename \"${CHECKPOINT_PATH}\" | sed 's/--.*//')\" \
    --local_log_dir \"${OUTPUT_DIR}\" \
    --save_rollout_video False \
    --seed 7"

# 运行评测
eval ${EVAL_CMD} 2>&1 | tee "${OUTPUT_DIR}/eval_$(basename "${CHECKPOINT_PATH}" | sed 's/--.*//')_$(date +%Y%m%d_%H%M%S).log"

EVAL_EXIT_CODE=$?

# 计算总时长
EVAL_END_TIME=$(date +%s)
EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
EVAL_MINUTES=$((EVAL_DURATION / 60))
EVAL_SECONDS=$((EVAL_DURATION % 60))

echo ""
echo "============================================"
if [ ${EVAL_EXIT_CODE} -eq 0 ]; then
    echo "✅ 评测完成！"
else
    echo "❌ 评测失败 (exit code: ${EVAL_EXIT_CODE})"
fi
echo "⏱️  总耗时: ${EVAL_MINUTES}分${EVAL_SECONDS}秒 (${EVAL_DURATION}秒)"
echo "============================================"
echo ""

# 显示结果摘要
echo "📊 结果摘要："
echo "============================================"
LATEST_LOG=$(ls -t "${OUTPUT_DIR}"/eval_*.log 2>/dev/null | head -1)
if [ -f "${LATEST_LOG}" ]; then
    echo "最新日志: ${LATEST_LOG}"
    echo ""
    echo "成功率统计:"
    grep "Overall success rate" "${LATEST_LOG}" || echo "未找到成功率统计"
    echo ""
    echo "各任务详细结果:"
    grep "Task " "${LATEST_LOG}" | grep "success rate" || echo "未找到任务详情"
else
    echo "未找到日志文件"
fi
echo "============================================"

