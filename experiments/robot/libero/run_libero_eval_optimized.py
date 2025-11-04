"""
run_libero_eval_optimized.py

优化版本的LIBERO评测脚本 - 改进GPU利用率

主要优化:
1. 移除model_lock，允许多GPU并行加载模型
2. 更智能的GPU分配策略
3. 减少模型加载等待时间
"""

import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from pathlib import Path
import time
from typing import Optional, Union, Any, Dict, List

import draccus
import imageio
from joblib import Parallel, delayed
import numpy as np
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from libero.libero import benchmark

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.openvla_utils import model_is_on_hf_hub, update_auto_map, check_model_logic_mismatch, _apply_film_to_vla, _load_dataset_stats

# Append current directory so that interpreter can find experiments.robot
sys.path.append("../..")
from experiments.robot.libero.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from experiments.robot.openvla_utils import (
    get_action_head,
    get_noisy_action_projector,
    get_processor,
    get_proprio_projector,
    resize_image_for_policy,
    prepare_images_for_vla,
    normalize_proprio,
)
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
    get_image_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)
from prismatic.vla.constants import NUM_ACTIONS_CHUNK


# 导入原始脚本的所有其他函数
from experiments.robot.libero.run_libero_eval import (
    TaskSuite,
    TASK_MAX_STEPS,
    GenerateConfig,
    log_message,
    validate_config,
    setup_logging,
    load_initial_states,
    get_model,
    initialize_model,
    get_observation,
    process_action,
    save_rollout_video,
    predict_action,
    run_episode,
)


def get_assigned_gpu(task_id: int, num_gpus: int) -> int:
    """
    智能GPU分配策略：轮询分配
    
    Args:
        task_id: 任务ID
        num_gpus: 可用GPU数量
    
    Returns:
        int: 分配的GPU ID
    """
    return task_id % num_gpus


def run_task_optimized(
    log_lock,
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    resize_size,
    log_file=None,
):
    """优化版本的任务运行函数 - 移除model_lock，允许并行加载"""
    
    # Set random seed
    set_seed_everywhere(cfg.seed+task_id)
    
    # 获取可用GPU数量
    num_gpus = torch.cuda.device_count()
    
    # 智能分配GPU（轮询策略）
    assigned_gpu = get_assigned_gpu(task_id, num_gpus)
    device = torch.device(f'cuda:{assigned_gpu}')
    
    log_message(f'Task {task_id}: Assigned to GPU {assigned_gpu}', log_file, log_lock)
    
    # 直接加载模型，无需等待锁
    start_time = time.time()
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg, device)
    load_time = time.time() - start_time
    log_message(f'Task {task_id}: Model loaded in {load_time:.2f}s on GPU {assigned_gpu}', log_file, log_lock)
    
    # Get task
    task = task_suite.get_task(task_id)
    
    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file, log_lock)
    
    # Initialize environment and get task description
    env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
    
    # Start episodes
    task_episodes, task_successes = 0, 0
    task_start_time = time.time()
    
    for episode_idx in range(cfg.num_trials_per_task):
        # Handle initial state
        if cfg.initial_states_path == "DEFAULT":
            initial_state = initial_states[episode_idx]
        else:
            initial_states_task_key = task_description.replace(" ", "_")
            episode_key = f"demo_{episode_idx}"
            
            if not all_initial_states[initial_states_task_key][episode_key]["success"]:
                log_message(f"Skipping task {task_id} episode {episode_idx} due to failed expert demo!", log_file, log_lock)
                continue
            
            initial_state = np.array(all_initial_states[initial_states_task_key][episode_key]["initial_state"])
        
        log_message(f"Task {task_id}: Starting episode {task_episodes + 1}...", log_file, log_lock)
        
        episode_start = time.time()
        
        # Run episode
        success, replay_images = run_episode(
            cfg,
            env,
            task_description,
            model,
            resize_size,
            processor,
            action_head,
            proprio_projector,
            noisy_action_projector,
            initial_state,
            log_file,
            log_lock,
            device,
        )
        
        episode_time = time.time() - episode_start
        
        # Update counters
        task_episodes += 1
        if success:
            task_successes += 1
        
        log_message(
            f"Task {task_id} Episode {task_episodes}: {'Success' if success else 'Failure'} "
            f"({episode_time:.1f}s, GPU {assigned_gpu})",
            log_file,
            log_lock
        )
        
        # Save rollout video
        if cfg.save_rollout_video or not success:
            save_rollout_video(
                cfg,
                replay_images,
                task_id,
                task_description,
                task_episodes - 1,
                success,
            )
    
    task_time = time.time() - task_start_time
    task_success_rate = task_successes / task_episodes if task_episodes > 0 else 0
    
    log_message(
        f"Task {task_id} completed: {task_successes}/{task_episodes} "
        f"({task_success_rate*100:.1f}%) in {task_time:.1f}s on GPU {assigned_gpu}",
        log_file,
        log_lock
    )
    
    # Cleanup
    del model, action_head, proprio_projector, noisy_action_projector, processor
    torch.cuda.empty_cache()
    
    return task_successes, task_episodes


@draccus.wrap()
def eval_libero_optimized(cfg: GenerateConfig) -> float:
    """优化版本的LIBERO评测主函数"""
    
    # Validate configuration
    validate_config(cfg)
    
    # Set random seed
    set_seed_everywhere(cfg.seed)
    
    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    
    # Setup logging
    log_file, run_id = setup_logging(cfg)
    
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks
    
    # 获取GPU信息
    num_gpus = torch.cuda.device_count()
    log_message(f"Available GPUs: {num_gpus}", log_file, None)
    log_message(f"Task suite: {cfg.task_suite_name} ({num_tasks} tasks)", log_file, None)
    log_message(f"Trials per task: {cfg.num_trials_per_task}", log_file, None)
    log_message(f"Total episodes: {num_tasks * cfg.num_trials_per_task}", log_file, None)
    log_message("="*80, log_file, None)
    log_message("🚀 OPTIMIZATION: Parallel model loading enabled (no model_lock)", log_file, None)
    log_message(f"🚀 OPTIMIZATION: GPU assignment strategy: round-robin across {num_gpus} GPUs", log_file, None)
    log_message("="*80, log_file, None)
    
    manager = multiprocessing.Manager()
    log_lock = manager.Lock()
    
    # Start evaluation - 使用优化版本的run_task
    # n_jobs=num_tasks 允许所有任务并行运行
    eval_start = time.time()
    
    results = Parallel(n_jobs=num_tasks)(
        delayed(run_task_optimized)(
            log_lock, cfg, task_suite, task_id, resize_size, log_file
        ) for task_id in range(num_tasks)
    )
    
    eval_time = time.time() - eval_start
    
    total_successes = sum(task[0] for task in results)
    total_episodes = sum(task[1] for task in results)
    
    # Calculate final success rate
    final_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0
    
    # Log final results
    log_message("="*80, log_file, log_lock)
    log_message("📊 Final Results:", log_file, log_lock)
    log_message(f"Total episodes: {total_episodes}", log_file, log_lock)
    log_message(f"Total successes: {total_successes}", log_file, log_lock)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file, log_lock)
    log_message(f"Total time: {eval_time:.1f}s ({eval_time/60:.1f} min)", log_file, log_lock)
    log_message(f"Avg time per episode: {eval_time/total_episodes:.1f}s", log_file, log_lock)
    log_message("="*80, log_file, log_lock)
    
    return final_success_rate


if __name__ == "__main__":
    eval_libero_optimized()

