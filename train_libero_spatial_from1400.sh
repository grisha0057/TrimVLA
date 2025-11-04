#!/bin/bash

# 从 1400 步继续训练，min_keep_ratio 0.40 → 0.30，线性 ramp 1400 步

# ========== 渲染配置 ==========
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# ========== Python 输出配置 ==========
export PYTHONUNBUFFERED=1

# ========== PyTorch 优化配置 ==========
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# ========== 路径配置 ==========
VLA_PATH=${VLA_PATH:-"/root/workspace/LightVLA/checkpoints/openvla-libero-spatial"}
DATA_ROOT_DIR=${DATA_ROOT_DIR:-"/root/workspace/LightVLA/datasets/rlds/modified_libero_rlds_full"}
DATASET_NAME=${DATASET_NAME:-"libero_spatial_no_noops"}
RUN_ROOT_DIR=${RUN_ROOT_DIR:-"/root/workspace/LightVLA/logs/libero_spatial_training"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"libero_spatial_from1400_$(date +%Y%m%d_%H%M%S)"}

# 如果你有 1400 步的 checkpoint，建议设置：START_CHECKPOINT=/path/to/run_id--1400_chkpt
START_CHECKPOINT=${START_CHECKPOINT:-"${VLA_PATH}"}

# ========== 训练超参数 ==========
LR=${LR:-1e-4}
WARMUP_STEPS=${WARMUP_STEPS:-0}          # LR warmup 关闭
SAVE_FREQ=${SAVE_FREQ:-200}
DECAY_MILESTONES=${DECAY_MILESTONES:-"[100000]"}
DECAY_GAMMA=${DECAY_GAMMA:-0.5}

# 继续训练 1400 步：1400 -> 2800
RESUME_START_STEP=${RESUME_START_STEP:-1400}
MAX_STEPS=${MAX_STEPS:-2800}

# 批次与梯度累积
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUMULATION=${GRAD_ACCUMULATION:-8}

# LoRA 配置
LORA_RANK=${LORA_RANK:-8}

# 数据增强
IMAGE_AUG=${IMAGE_AUG:-False}

# ========== 筛选与调度（关键） ==========
PRUNE_DISABLE=${PRUNE_DISABLE:-False}
PRUNE_DISABLE_KEEP_BINS=${PRUNE_DISABLE_KEEP_BINS:-True}

# Coverage 跟随 min_keep_ratio，略高出 offset
PRUNE_COVERAGE_FOLLOW_MIN_KEEP=${PRUNE_COVERAGE_FOLLOW_MIN_KEEP:-True}
PRUNE_COVERAGE_OFFSET=${PRUNE_COVERAGE_OFFSET:-0.05}
COVERAGE_WARMUP=${COVERAGE_WARMUP:-0.40}

# 聚合与 rescale
PRUNE_AGGREGATION=${PRUNE_AGGREGATION:-"logsumexp"}
PRUNE_LSE_TEMP=${PRUNE_LSE_TEMP:-1.0}
PRUNE_RESCALE=${PRUNE_RESCALE:-True}
PRUNE_CLIP=${PRUNE_CLIP:-10.0}

# ST‑TopK 训练（tau 保持常数 0.30）
PRUNE_TRAIN_USE_ST_TOPK=${PRUNE_TRAIN_USE_ST_TOPK:-True}
PRUNE_TAU_START=${PRUNE_TAU_START:-0.30}
PRUNE_TAU_END=${PRUNE_TAU_END:-0.30}
PRUNE_TAU_RAMP_STEPS=${PRUNE_TAU_RAMP_STEPS:-1}

# 最小保留比例：从 0.38 线性降到 0.30，用时 1400 步
# 注意：不用 prune_schedule_warmup_step，改用“相对步数”控制（见下方 resume_step 逻辑）
PRUNE_MIN_KEEP_RATIO_WARMUP=${PRUNE_MIN_KEEP_RATIO_WARMUP:-0.39}
PRUNE_MIN_KEEP_RATIO_TARGET=${PRUNE_MIN_KEEP_RATIO_TARGET:-0.30}
PRUNE_MIN_KEEP_RAMP_STEPS=${PRUNE_MIN_KEEP_RAMP_STEPS:-1400}

# ========== 评估配置 ==========
EVAL_NUM_TRIALS=${EVAL_NUM_TRIALS:-5}
EVAL_GPUS=${EVAL_GPUS:-"0,1"}

# ========== 分布式 ==========
NPROC=${NPROC:-2}
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export CUDA_VISIBLE_DEVICES=${CUDA_DEVICES}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "============================================"
echo "🚀 From 1400: LightVLA - LIBERO Spatial 续训 + 评估"
echo "============================================"
echo "  - 起始 checkpoint: ${START_CHECKPOINT} (resume_step=${RESUME_START_STEP})"
echo "  - 训练至: ${MAX_STEPS} 步（共 $((MAX_STEPS-RESUME_START_STEP)) 步）"
echo "  - 保存/评估频率: 每 ${SAVE_FREQ} 步"
echo "  - min_keep_ratio: ${PRUNE_MIN_KEEP_RATIO_WARMUP} -> ${PRUNE_MIN_KEEP_RATIO_TARGET} (ramp ${PRUNE_MIN_KEEP_RAMP_STEPS})"
echo "  - tau: ${PRUNE_TAU_START} -> ${PRUNE_TAU_END} (ramp ${PRUNE_TAU_RAMP_STEPS})"
echo "  - Coverage: 跟随 min_keep_ratio (+${PRUNE_COVERAGE_OFFSET})"
echo ""

# 基础检查
if [ ! -d "${START_CHECKPOINT}" ]; then
  echo "❌ 错误: 起始 checkpoint 路径不存在: ${START_CHECKPOINT}"
  exit 1
fi
if [ ! -d "${DATA_ROOT_DIR}/${DATASET_NAME}" ]; then
  echo "❌ 错误: 数据集不存在: ${DATA_ROOT_DIR}/${DATASET_NAME}"
  exit 1
fi

EXPERIMENT_DIR="${RUN_ROOT_DIR}/${EXPERIMENT_NAME}"
mkdir -p ${EXPERIMENT_DIR}
LOG_FILE="${EXPERIMENT_DIR}/train.log"
EVAL_LOG_FILE="${EXPERIMENT_DIR}/eval_results.log"
RUN_ID_FIXED="${EXPERIMENT_NAME}"

# 保存配置
cat > ${EXPERIMENT_DIR}/config.txt <<EOF
续训配置 - $(date)
==================
起始 checkpoint: ${START_CHECKPOINT}
resume_step: ${RESUME_START_STEP}
训练至: ${MAX_STEPS}
保存/评估频率: ${SAVE_FREQ}

min_keep_ratio: ${PRUNE_MIN_KEEP_RATIO_WARMUP} -> ${PRUNE_MIN_KEEP_RATIO_TARGET}
ramp_steps: ${PRUNE_MIN_KEEP_RAMP_STEPS}
coverage_follow_min_keep: ${PRUNE_COVERAGE_FOLLOW_MIN_KEEP} (+${PRUNE_COVERAGE_OFFSET})
tau: ${PRUNE_TAU_START} -> ${PRUNE_TAU_END} (ramp ${PRUNE_TAU_RAMP_STEPS})
EOF

# 修复主机名解析
HN=$(hostname)
if ! grep -q "\b${HN}\b" /etc/hosts 2>/dev/null; then
  echo "🧩 修复主机名解析..."
  {
    echo "127.0.0.1 ${HN}"
  } >> /etc/hosts 2>/dev/null || echo "⚠️ 无法修改 /etc/hosts"
fi

# 评估函数：根据 step 映射到日程中的 min_keep_ratio
run_evaluation() {
  local checkpoint_path=$1
  local step=$2

  local warm_abs=${RESUME_START_STEP}
  local r0=${PRUNE_MIN_KEEP_RATIO_WARMUP}
  local r1=${PRUNE_MIN_KEEP_RATIO_TARGET}
  local ramp=${PRUNE_MIN_KEEP_RAMP_STEPS}
  local ratio
  if [ ${step} -lt ${warm_abs} ]; then
    ratio=${r0}
  else
    local passed=$(( step - warm_abs ))
    local p=$(python -c "passed=${passed};rr=${ramp};print(min(1.0, max(0.0, passed/float(max(1, rr)))))")
    ratio=$(python -c "rw=${r0};rt=${r1};p=${p};print((1.0-p)*rw + p*rt)")
  fi

  local cov
  if [ "${PRUNE_COVERAGE_FOLLOW_MIN_KEEP}" = "True" ]; then
    cov=$(python -c "ratio=${ratio};off=${PRUNE_COVERAGE_OFFSET};print(min(0.999, ratio+off))")
  else
    cov=${COVERAGE_WARMUP}
  fi

  echo ""
  echo "============================================"
  echo "🎮 开始评估 Checkpoint: ${checkpoint_path}"
  echo "   Step: ${step} | min_keep_ratio=${ratio} | coverage=${cov}"
  echo "============================================"

  local TRAIN_CUDA_DEVICES=${CUDA_VISIBLE_DEVICES}
  export CUDA_VISIBLE_DEVICES=${EVAL_GPUS}

  local eval_start_time=$(date +%s)
  cd /root/workspace/LightVLA
  python -u experiments/robot/libero/run_libero_eval.py \
    --pretrained_checkpoint "${checkpoint_path}" \
    --task_suite_name "libero_spatial" \
    --use_l1_regression True \
    --use_diffusion False \
    --use_film False \
    --num_images_in_input 2 \
    --use_proprio True \
    --lora_rank ${LORA_RANK} \
    --center_crop False \
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
  export CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_DEVICES}
  if [ ${eval_exit_code} -eq 0 ]; then
    echo "✅ 评估完成 (耗时: ${eval_duration}秒)" | tee -a ${EVAL_LOG_FILE}
  else
    echo "❌ 评估失败 (exit code: ${eval_exit_code})" | tee -a ${EVAL_LOG_FILE}
  fi
}

# ========== 单次训练 + 训练后批量评估（1400 步） ==========
cd /root/workspace/LightVLA

CURRENT_STEP=${RESUME_START_STEP}
LAST_CHECKPOINT_PATH="${START_CHECKPOINT}"

REMAIN_STEPS=$(( MAX_STEPS - CURRENT_STEP ))
if [ ${REMAIN_STEPS} -le 0 ]; then
  echo "❌ MAX_STEPS(${MAX_STEPS}) 必须大于 RESUME_START_STEP(${CURRENT_STEP})"
  exit 1
fi
NUM_STAGES=$(( REMAIN_STEPS / SAVE_FREQ ))
echo "📝 续训计划: ${CURRENT_STEP} -> ${MAX_STEPS}，共 ${REMAIN_STEPS} 步（每 ${SAVE_FREQ} 步保存，训练结束后逐个评估）"

echo ""
echo "============================================"
echo "🏃 开始单次训练（相对步数 0 -> ${REMAIN_STEPS}）"
echo "   起始 checkpoint: ${LAST_CHECKPOINT_PATH}"
echo "============================================"

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
  --resume False \
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
  --max_steps ${REMAIN_STEPS} \
  --save_freq ${SAVE_FREQ} \
  --save_latest_checkpoint_only False \
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
  echo "❌ 训练失败 (exit code: ${TRAIN_EXIT_CODE})"
  exit ${TRAIN_EXIT_CODE}
fi

echo "✅ 单次训练完成"

# 训练结束后批量评估：按照保存的相对步数 200, 400, ... 依次评估
echo ""
echo "============================================"
echo "🔎 开始训练后批量评估"
echo "============================================"

# 列出所有保存的 checkpoint 目录并按相对步数排序
CKPTS=( $(find "${EXPERIMENT_DIR}" -maxdepth 1 -type d -name "*--*_chkpt" | sed -E 's@.*/--([0-9]+)_chkpt$@\1 \0@' | sort -n | awk '{print $2}') )

if [ ${#CKPTS[@]} -eq 0 ]; then
  echo "⚠️ 未找到任何 checkpoint 目录 (*--*_chkpt)"
else
  for ck in "${CKPTS[@]}"; do
    # 提取相对步数并映射到绝对步数
    rel=$(basename "${ck}" | sed -E 's/.*--([0-9]+)_chkpt/\1/')
    abs=$(( RESUME_START_STEP + rel ))
    run_evaluation "${ck}" "${abs}"
  done
fi

echo ""
echo "============================================"
echo "🎉 续训+评估完成 (${RESUME_START_STEP} -> ${MAX_STEPS})"
echo "📂 实验目录: ${EXPERIMENT_DIR}"
echo "📝 训练日志: ${LOG_FILE}"
echo "📊 评估日志: ${EVAL_LOG_FILE}"
echo "============================================"
