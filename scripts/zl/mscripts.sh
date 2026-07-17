# /bin/bash
# 仿真
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=/data/users/zhaolei/projs/core/Isaac-GR00T:$PYTHONPATH python scripts/gr00t_finetune.py \
--dataset-path /mnt/raid/zhaolei/dataset/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/gr1_arms_waist.CupToDrawer \
--output-dir /mnt/raid/zhaolei/output/gr00t/add_dinov2_simple_fusion_sim_CupToDrawer_1k \
--base_model_path /data/shared_checkpoints/gr00t_pretrain/GR00T-N1.5-3B-new \
--num-gpus 8 \
--batch-size 64 \
--learning_rate 3e-5 \
--data-config fourier_gr1_arms_waist \
--embodiment_tag gr1 \
--max-steps 60000 \
--save-steps 3000

# 真机
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONPATH=/data/users/zhaolei/projs/core/Isaac-GR00T:$PYTHONPATH python scripts/gr00t_finetune.py \
--dataset-path /data/shared_datasets/microwave/task_287 /data/shared_datasets/microwave/task_288_400/task_288 \
--output-dir /mnt/raid/zhaolei/output/gr00t/add_dinov2_simple_fusion \
--base_model_path /data/shared_checkpoints/gr00t_pretrain/GR00T-N1.5-3B-new \
--num-gpus 8 \
--batch-size 32 \
--max-steps 60000 \
--save-steps 3000

# 离线推理
PYTHONPATH=/data/users/zhaolei/projs/core/Isaac-GR00T:$PYTHONPATH python scripts/inference_service.py --server \
--model_path /mnt/raid/zhaolei/output/gr00t/add_dinov2_simple_fusion/checkpoint-60000 \
--embodiment_tag a2d \
--data_config a2d_arms_only

PYTHONPATH==/data/users/zhaolei/projs/core/Isaac-GR00T:$PYTHONPATH python scripts/eval_policy.py --plot \
--dataset_path /data/shared_datasets/microwave/task_288_400/task_288 \
--embodiment_tag a2d \
--modality_keys left_arm left_hand right_arm right_hand \
--steps 1000 --trajs 1 --data_config a2d_arms_only --action_horizon 64 \
--save_plot_path /mnt/raid/zhaolei/output/gr00t/add_dinov2_simple_fusion/checkpoint-60000/eval.png

# 仿真离线推理
PYTHONPATH=$PWD/gr00t:$PYTHONPATH python scripts/inference_service.py --server \
--model_path $PWD/../checkpoint-30000 \
--data_config fourier_gr1_arms_waist \
--embodiment_tag gr1

PYTHONPATH=$PWD/gr00t:$PYTHONPATH python scripts/simulation_service.py --client \
--env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
--video_dir $PWD/../videos \
--max_episode_steps 720 \
--n_envs 5 \
--n_episodes 50
