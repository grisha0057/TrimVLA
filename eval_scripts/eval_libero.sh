python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "path/to/lightvla_10_40000_chkpt" \
  --task_suite_name libero_10 \
  --save_rollout_video false \
  --num_trials_per_task 50
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "path/to/lightvla_goal_40000_chkpt" \
  --task_suite_name libero_goal \
  --save_rollout_video false \
  --num_trials_per_task 50
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "path/to/lightvla_object_40000_chkpt" \
  --task_suite_name libero_object \
  --save_rollout_video false \
  --num_trials_per_task 50
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "path/to/lightvla_spatial_40000_chkpt" \
  --task_suite_name libero_spatial \
  --save_rollout_video false \
  --num_trials_per_task 50
