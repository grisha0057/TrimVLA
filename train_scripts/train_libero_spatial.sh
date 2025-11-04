#!/bin/bash

# ========== 渲染配置 ==========
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# ========== Python 输出配置 ==========
# 禁用 Python 输出缓冲，让日志实时显示
export PYTHONUNBUFFERED=1

# ========== PyTorch 优化配置 ==========
# 启用 TF32 加速（Ampere及以上架构）
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# ========== 路径配置 ==========
VLA_PATH=${VLA_PATH:-"/root/workspace/LightVLA/checkpoints/openvla-libero-spatial"}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-"/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds_full"}
DATASET_NAME=${DATASET_NAME:-"libero_spatial_no_noops"}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-"/root/workspace/LightVLA/logs/libero_spatial_training"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"libero_spatial_$(date +%Y%m%d_%H%M%S)"}

# ========== 训练超参数（与 overfit 实验保持一致）==========
# 学习率与调度
LR=${LR:-1e-4}
MAX_STEPS=${MAX_STEPS:-1400}
WARMUP_STEPS=${WARMUP_STEPS:-0}
DECAY_MILESTONES=${DECAY_MILESTONES:-"[100000]"}  # 基本不衰减
DECAY_GAMMA=${DECAY_GAMMA:-0.5}

# 批次与梯度累积（优化后：提升GPU利用率）
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:-8}

# LoRA 配置（与 overfit 一致）
LORA_RANK=${LORA_RANK:-8}

# 保存策略
SAVE_FREQ=${SAVE_FREQ:-200}
SAVE_LATEST_ONLY=${SAVE_LATEST_ONLY:-False}

IMAGE_AUG=${IMAGE_AUG:-True}

# 禁用筛选
PRUNE_DISABLE=${PRUNE_DISABLE:-False}

# Coverage 参数（副旋钮）：跟随最小保留
COVERAGE_WARMUP=${COVERAGE_WARMUP:-1.0}
PRUNE_COVERAGE_FOLLOW_MIN_KEEP=${PRUNE_COVERAGE_FOLLOW_MIN_KEEP:-True}
PRUNE_COVERAGE_OFFSET=${PRUNE_COVERAGE_OFFSET:-0.05}

# 聚合方式：logsumexp（推荐）| mean | max
PRUNE_AGGREGATION=${PRUNE_AGGREGATION:-"logsumexp"}
PRUNE_LSE_TEMP=${PRUNE_LSE_TEMP:-1.0}      # LogSumExp 温度参数

# Soft rescale 参数
PRUNE_RESCALE=${PRUNE_RESCALE:-True}        # 启用均值保持的 rescale
PRUNE_CLIP=${PRUNE_CLIP:-10.0}             # Rescale 裁剪阈值

# ST-TopK 训练（Gumbel-Softmax + 直通）
PRUNE_TRAIN_USE_ST_TOPK=${PRUNE_TRAIN_USE_ST_TOPK:-True}
PRUNE_TAU_START=${PRUNE_TAU_START:-1.0}
PRUNE_TAU_END=${PRUNE_TAU_END:-0.30}
PRUNE_TAU_RAMP_STEPS=${PRUNE_TAU_RAMP_STEPS:-1200}

# 最小保留比例（主旋钮）
PRUNE_DISABLE_KEEP_BINS=${PRUNE_DISABLE_KEEP_BINS:-True}
PRUNE_MIN_KEEP_RATIO_WARMUP=${PRUNE_MIN_KEEP_RATIO_WARMUP:-1.0}
PRUNE_MIN_KEEP_RATIO_TARGET=${PRUNE_MIN_KEEP_RATIO_TARGET:-0.20}
PRUNE_MIN_KEEP_RAMP_STEPS=${PRUNE_MIN_KEEP_RAMP_STEPS:-1200}

# ========== 评估配置 ==========
EVAL_NUM_TRIALS=${EVAL_NUM_TRIALS:-3}       # 每个任务评估3次（快速评估）
EVAL_GPUS=${EVAL_GPUS:-"0,1"}               # 评估使用的GPU

# ========== 分布式训练配置 ==========
# GPU 配置
NPROC=${NPROC:-2}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}

# 网络配置
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}


# ========== 前置检查 ==========
echo "============================================"
echo "🚀 LightVLA - LIBERO Spatial 训练+评估"
echo "============================================"
echo ""
echo "📊 训练配置（优化版：提升训练速度）："
echo "  - 模型路径: ${VLA_PATH}"
echo "  - 数据路径: ${DATA_ROOT_DIR}"
echo "  - 数据集名称: ${DATASET_NAME}"
echo "  - 实验名称: ${EXPERIMENT_NAME}"
echo ""
echo "🎯 训练超参数："
echo "  - 学习率: ${LR}"
echo "  - 总步数: ${MAX_STEPS}"
echo "  - 保存频率: 每 ${SAVE_FREQ} 步"
echo "  - Warmup: ${WARMUP_STEPS} 步"
echo "  - 批次大小: ${BATCH_SIZE}"
echo "  - 梯度累积: ${GRAD_ACCUMULATION}"
echo "  - LoRA Rank: ${LORA_RANK}"
echo "  - 图像增强: ${IMAGE_AUG}"
echo ""
echo "🔍 视觉 Token 筛选："
echo "  - 启用: $([ "${PRUNE_DISABLE}" = "False" ] && echo '✅' || echo '❌')"
echo "  - 最小保留比例: ${PRUNE_MIN_KEEP_RATIO_WARMUP} -> ${PRUNE_MIN_KEEP_RATIO_TARGET} (ramp ${PRUNE_MIN_KEEP_RAMP_STEPS})"
echo "  - Coverage: 跟随 min_keep_ratio (follow_min_keep=${PRUNE_COVERAGE_FOLLOW_MIN_KEEP}, +${PRUNE_COVERAGE_OFFSET})"
echo "  - 量化桶: $([ "${PRUNE_DISABLE_KEEP_BINS}" = "True" ] && echo '禁用' || echo '启用')"
echo "  - Gumbel 温度: ${PRUNE_TAU_START} -> ${PRUNE_TAU_END} (ramp ${PRUNE_TAU_RAMP_STEPS})"
echo "  - 聚合方式: ${PRUNE_AGGREGATION}"
echo ""
echo "🎮 评估配置："
echo "  - 评估频率: 每 ${SAVE_FREQ} 步"
echo "  - 每任务试验次数: ${EVAL_NUM_TRIALS}"
echo "  - 评估GPU: ${EVAL_GPUS}"
echo ""
echo "🖥️  训练GPU: ${CUDA_DEVICES} (${NPROC}卡)"
echo ""

# 检查模型
if [ ! -d "${VLA_PATH}" ]; then
    echo "❌ 错误: 模型路径不存在: ${VLA_PATH}"
    exit 1
fi

# 检查数据集
if [ ! -d "${DATA_ROOT_DIR}/${DATASET_NAME}" ]; then
    echo "❌ 错误: 数据集不存在: ${DATA_ROOT_DIR}/${DATASET_NAME}"
    exit 1
fi

# 检查 dataset_info.json
DATASET_INFO="${DATA_ROOT_DIR}/${DATASET_NAME}/1.0.0/dataset_info.json"
if [ -f "${DATASET_INFO}" ]; then
    echo "✅ 数据集信息："
    if command -v jq >/dev/null 2>&1; then
        NUM_SHARDS=$(jq '.splits[0].shardLengths | length' "${DATASET_INFO}")
        SHARD_LENGTH=$(jq '.splits[0].shardLengths[0]' "${DATASET_INFO}")
        TOTAL_SAMPLES=$((NUM_SHARDS * SHARD_LENGTH))
        NUM_BYTES=$(jq -r '.splits[0].numBytes' "${DATASET_INFO}")
        echo "  - 分片数: ${NUM_SHARDS}"
        echo "  - 每分片样本: ${SHARD_LENGTH}"
        echo "  - 总样本数: ${TOTAL_SAMPLES}"
        echo "  - 数据大小: $((NUM_BYTES / 1024 / 1024)) MB"
    fi
fi

echo ""
echo "✅ 检查通过"
echo ""

# ========== 创建实验目录 ==========
EXPERIMENT_DIR="${RUN_ROOT_DIR}/${EXPERIMENT_NAME}"
mkdir -p ${EXPERIMENT_DIR}
LOG_FILE="${EXPERIMENT_DIR}/train.log"
EVAL_LOG_FILE="${EXPERIMENT_DIR}/eval_results.log"

# 固定 run_id，避免续训时名称越来越长；与 finetune 的 --run_id_override 对齐
RUN_ID_FIXED="${EXPERIMENT_NAME}"

# 保存配置
cat > ${EXPERIMENT_DIR}/config.txt <<EOF
训练+评估配置 - $(date)
==================
模型: ${VLA_PATH}
数据集: ${DATA_ROOT_DIR}/${DATASET_NAME}
实验名称: ${EXPERIMENT_NAME}

训练超参数:
  学习率: ${LR}
  总步数: ${MAX_STEPS}
  保存频率: ${SAVE_FREQ}
  Warmup: ${WARMUP_STEPS}
  批次大小: ${BATCH_SIZE}
  梯度累积: ${GRAD_ACCUMULATION}
  LoRA Rank: ${LORA_RANK}
  图像增强: ${IMAGE_AUG}

视觉 Token 筛选:
  禁用: ${PRUNE_DISABLE}
  Coverage: 跟随 min_keep_ratio (+${PRUNE_COVERAGE_OFFSET})
  聚合: ${PRUNE_AGGREGATION}
  温度: ${PRUNE_LSE_TEMP}
  Rescale: ${PRUNE_RESCALE}

评估配置:
  频率: 每 ${SAVE_FREQ} 步
  每任务试验: ${EVAL_NUM_TRIALS}
  评估GPU: ${EVAL_GPUS}

训练GPU: ${CUDA_DEVICES}
EOF

# ========== 修复主机名解析 ==========
HN=$(hostname)
if ! grep -q "\b${HN}\b" /etc/hosts 2>/dev/null; then
  echo "🧩 修复主机名解析..."
  {
    echo "127.0.0.1 ${HN}"
  } >> /etc/hosts 2>/dev/null || echo "⚠️ 无法修改 /etc/hosts"
fi

# ========== 定义评估函数 ==========
run_evaluation() {
    local checkpoint_path=$1
    local step=$2
    
    # 根据当前 step 计算评测用的 min_keep_ratio 与 coverage
    local ratio_warm=${PRUNE_MIN_KEEP_RATIO_WARMUP}
    local ratio_tgt=${PRUNE_MIN_KEEP_RATIO_TARGET}
    local ratio_ramp=${PRUNE_MIN_KEEP_RAMP_STEPS}
    local cov_follow=${PRUNE_COVERAGE_FOLLOW_MIN_KEEP}
    local cov_off=${PRUNE_COVERAGE_OFFSET}
    local ratio
    if [ ${step} -lt ${WARMUP_STEPS} ]; then
        ratio=${ratio_warm}
    else
        # 线性插值：从 warmup 步开始到 ramp 完成
        local passed=$(( step - WARMUP_STEPS ))
        if [ ${passed} -lt 0 ]; then passed=0; fi
        if [ ${ratio_ramp} -le 0 ]; then
            ratio=${ratio_tgt}
        else
            # clamp 到 [0,1]
            local num=$(python - <<PY
passed=${passed}
ratio_ramp=${ratio_ramp}
print(min(1.0, max(0.0, passed/ratio_ramp)))
PY
)
            ratio=$(python - <<PY
rw=${ratio_warm}
rt=${ratio_tgt}
p=${num}
print((1.0-p)*rw + p*rt)
PY
)
        fi
    fi
    local cov
    if [ "${cov_follow}" = "True" ]; then
        cov=$(python - <<PY
ratio=${ratio}
off=${cov_off}
print(min(0.999, ratio+off))
PY
)
    else
        echo "❌ 错误: PRUNE_COVERAGE_FOLLOW_MIN_KEEP 必须为 True"
        exit 1
    fi
    
    echo ""
    echo "============================================"
    echo "🎮 开始评估 Checkpoint: ${checkpoint_path}"
    echo "   Step: ${step} | min_keep_ratio=${ratio} | coverage=${cov}"
    echo "============================================"
    
    # 保存当前 CUDA_VISIBLE_DEVICES
    local TRAIN_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES}
    
    # 设置评估用的GPU
    export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}
    
    # 运行评估
    local eval_start_time=$(date +%s)
    
    cd /root/workspace/LightVLA
    
    # 使用 python -u 启用unbuffered输出，通过 tee 同时输出到控制台和文件
    python -u experiments/robot/libero/run_libero_eval.py \
        --pretrained_checkpoint "${checkpoint_path}" \
        --task_suite_name "libero_spatial" \
        --use_l1_regression True \
        --use_diffusion False \
        --use_film False \
        --num_images_in_input 2 \
        --use_proprio True \
        --lora_rank ${LORA_RANK} \
        --center_crop True \
        --num_trials_per_task ${EVAL_NUM_TRIALS} \
        --prune_disable_keep_bins ${PRUNE_DISABLE_KEEP_BINS} \
        --prune_min_keep_ratio ${ratio} \
        --prune_coverage_follow_min_keep ${PRUNE_COVERAGE_FOLLOW_MIN_KEEP} \
        --prune_coverage_offset ${PRUNE_COVERAGE_OFFSET} \
        --run_id_note "step_${step}" \
        --local_log_dir "${EXPERIMENT_DIR}/eval_logs" \
        --save_rollout_video False \
        --seed 7 2>&1 | tee -a ${EVAL_LOG_FILE}
    
    local eval_exit_code=$?
    local eval_end_time=$(date +%s)
    local eval_duration=$((eval_end_time - eval_start_time))
    
    # 恢复训练用的GPU设置
    export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES}
    
    if [ ${eval_exit_code} -eq 0 ]; then
        echo "✅ 评估完成 (耗时: ${eval_duration}秒)" | tee -a ${EVAL_LOG_FILE}
    else
        echo "❌ 评估失败 (exit code: ${eval_exit_code})" | tee -a ${EVAL_LOG_FILE}
    fi
    
    echo "============================================"
    echo ""
}

# ========== 分阶段训练+评估 ==========
cd /root/workspace/LightVLA

# 计算需要训练的阶段数
NUM_STAGES=$((MAX_STEPS / SAVE_FREQ))

echo "📝 训练计划: 总共 ${MAX_STEPS} 步，分 ${NUM_STAGES} 个阶段"
echo "   每阶段 ${SAVE_FREQ} 步后进行评估"
echo ""

# ========== Step 0: 评估初始模型（已跳过）==========
echo "============================================"
echo "⏭️  Step 0: 跳过初始模型评估（之前已评测）"
echo "============================================"
echo ""

CURRENT_STEP=0
LAST_CHECKPOINT_PATH="${VLA_PATH}"

for stage in $(seq 1 ${NUM_STAGES}); do
    TARGET_STEP=$((stage * SAVE_FREQ))
    STEPS_THIS_STAGE=$((TARGET_STEP - CURRENT_STEP))
    
    # 判断是否需要续训
    if [ ${stage} -eq 1 ]; then
        # 第一阶段：从头开始训练
        RESUME_FLAG="False"
        RESUME_STEP_ARG=""
        TRAIN_MODE="从头训练"
    else
        # 后续阶段：从上一个checkpoint继续训练
        RESUME_FLAG="True"
        RESUME_STEP_ARG="--resume_step ${CURRENT_STEP}"
        TRAIN_MODE="续训（从Step ${CURRENT_STEP}继续）"
    fi
    
    echo ""
    echo "============================================"
    echo "🏃 阶段 ${stage}/${NUM_STAGES}: 训练至 ${TARGET_STEP} 步"
    echo "   当前步数: ${CURRENT_STEP}"
    echo "   本阶段训练: ${STEPS_THIS_STAGE} 步"
    echo "   训练模式: ${TRAIN_MODE}"
    echo "   Checkpoint: ${LAST_CHECKPOINT_PATH}"
    echo "============================================"
    echo ""
    
    # 训练这一阶段
    # Python unbuffered 输出已通过 PYTHONUNBUFFERED=1 环境变量设置
    torchrun \
        --standalone \
        --nnodes 1 \
        --nproc-per-node ${NPROC} \
        --max-restarts 0 \
        vla-scripts/finetune.py \
        --vla_path "${LAST_CHECKPOINT_PATH}" \
        --data_root_dir "${DATA_ROOT_DIR}" \
        --dataset_name "${DATASET_NAME}" \
        --run_root_dir "${EXPERIMENT_DIR}" \
        --run_id_override "${RUN_ID_FIXED}" \
        --resume ${RESUME_FLAG} \
        ${RESUME_STEP_ARG} \
        --use_l1_regression True \
        --use_diffusion False \
        --use_film False \
        --num_images_in_input 2 \
        --use_proprio True \
        --batch_size ${BATCH_SIZE} \
        --grad_accumulation_steps ${GRAD_ACCUMULATION} \
        --learning_rate ${LR} \
        --lr_warmup_steps ${WARMUP_STEPS} \
        --lr_decay_milestones ${DECAY_MILESTONES} \
        --lr_decay_gamma ${DECAY_GAMMA} \
        --max_steps ${TARGET_STEP} \
        --save_freq ${SAVE_FREQ} \
        --save_latest_checkpoint_only ${SAVE_LATEST_ONLY} \
        --image_aug ${IMAGE_AUG} \
        --lora_rank ${LORA_RANK} \
        --prune_disable ${PRUNE_DISABLE} \
        --prune_coverage_warmup ${COVERAGE_WARMUP} \
        --prune_coverage_follow_min_keep ${PRUNE_COVERAGE_FOLLOW_MIN_KEEP} \
        --prune_coverage_offset ${PRUNE_COVERAGE_OFFSET} \
        --prune_prompt_aggregation ${PRUNE_AGGREGATION} \
        --prune_logsumexp_temperature ${PRUNE_LSE_TEMP} \
        --prune_soft_rescale_mean_preserve ${PRUNE_RESCALE} \
        --prune_soft_rescale_clip ${PRUNE_CLIP} \
        --prune_disable_keep_bins ${PRUNE_DISABLE_KEEP_BINS} \
        --prune_min_keep_ratio_warmup ${PRUNE_MIN_KEEP_RATIO_WARMUP} \
        --prune_min_keep_ratio_target ${PRUNE_MIN_KEEP_RATIO_TARGET} \
        --prune_min_keep_ramp_steps ${PRUNE_MIN_KEEP_RAMP_STEPS} \
        --prune_train_use_st_topk ${PRUNE_TRAIN_USE_ST_TOPK} \
        --prune_train_gumbel_tau_start ${PRUNE_TAU_START} \
        --prune_train_gumbel_tau_end ${PRUNE_TAU_END} \
        --prune_train_gumbel_tau_ramp_steps ${PRUNE_TAU_RAMP_STEPS} \
        --shuffle_buffer_size 10000 \
        --log_freq 10 2>&1 | tee -a ${LOG_FILE}
    
    TRAIN_EXIT_CODE=$?
    
    if [ ${TRAIN_EXIT_CODE} -ne 0 ]; then
        echo "❌ 训练阶段 ${stage} 失败 (exit code: ${TRAIN_EXIT_CODE})"
        exit ${TRAIN_EXIT_CODE}
    fi
    
    echo "✅ 阶段 ${stage} 训练完成"
    
    # 找到新生成的 checkpoint（包含时间戳的目录名）
    # finetune.py 生成的目录格式：run_id--${step}_chkpt
    # 我们需要找到包含 "--${TARGET_STEP}_chkpt" 的最新目录
    NEW_CHECKPOINT=$(find "${EXPERIMENT_DIR}" -maxdepth 1 -type d -name "*--${TARGET_STEP}_chkpt" | sort -r | head -1)
    
    if [ -z "${NEW_CHECKPOINT}" ] || [ ! -d "${NEW_CHECKPOINT}" ]; then
        echo "⚠️ 未找到 checkpoint: *--${TARGET_STEP}_chkpt"
        echo "   在目录: ${EXPERIMENT_DIR}"
        echo "   跳过本次评估"
        echo ""
        echo "   可用的checkpoint目录："
        ls -ld "${EXPERIMENT_DIR}"/*chkpt 2>/dev/null || echo "   无"
    else
        echo "✅ 找到 checkpoint: $(basename ${NEW_CHECKPOINT})"
        
        # 运行评估
        run_evaluation "${NEW_CHECKPOINT}" "${TARGET_STEP}"
        
        # 更新为下一阶段的起点
        LAST_CHECKPOINT_PATH="${NEW_CHECKPOINT}"
        echo "📝 下一阶段将从此checkpoint继续训练"
    fi
    
    # 更新当前步数
    CURRENT_STEP=${TARGET_STEP}
    
    echo ""
done

# ========== 训练结束 ==========
echo ""
echo "============================================"
echo "🎉 训练+评估流程全部完成！"
echo "============================================"
echo ""
echo "📂 实验目录: ${EXPERIMENT_DIR}"
echo "📝 训练日志: ${LOG_FILE}"
echo "📊 评估日志: ${EVAL_LOG_FILE}"
echo ""
echo "💾 Checkpoints:"
find "${EXPERIMENT_DIR}" -maxdepth 1 -type d -name "*chkpt" | sort
echo ""
echo "📈 评估结果汇总:"
if [ -f "${EVAL_LOG_FILE}" ]; then
    grep "Overall success rate" ${EVAL_LOG_FILE} || echo "未找到评估结果"
fi
echo ""
