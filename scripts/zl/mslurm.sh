#!/bin/bash

#SBATCH --job-name=heda_tea
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --output=slurm_logs/job_%j.out

export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
echo "MASTER_ADDR: $MASTER_ADDR MASTER_PORT: $MASTER_PORT"

export IS_TORCHRUN=1  # 该变量为gr00t特有
export GPU_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
export PYTHONPATH=/data/users/zhaolei/projs/core/Isaac-GR00T-N1.6:$PYTHONPATH

eval "$(conda shell.bash hook)"
conda activate gr00t_n16_zl

DATASET_BASE="/data/shared_datasets/heda2025/lerobot/pour_tea_to_cup"
DATASET_PATH=(
    # "$DATASET_BASE/pour_black_tea/task_404"
    # "$DATASET_BASE/pour_green_tea/task_405"
    # "$DATASET_BASE/pour_black_tea/task_413"
    # "$DATASET_BASE/pour_green_tea/task_414"
    # "$DATASET_BASE/pour_black_tea/task_420"
    # "$DATASET_BASE/pour_green_tea/task_421"
    # "$DATASET_BASE/pour_black_tea/task_431"
    # "$DATASET_BASE/pour_green_tea/task_432"
    # "$DATASET_BASE/pour_black_tea/task_433"
    # "$DATASET_BASE/pour_green_tea/task_434"
    # "$DATASET_BASE/pour_black_tea/task_456"
    # "$DATASET_BASE/pour_green_tea/task_458"
    # "$DATASET_BASE/pour_black_tea/task_469"
    # "$DATASET_BASE/pour_green_tea/task_470"
    # "$DATASET_BASE/pour_black_tea/task_471"
    # "$DATASET_BASE/pour_green_tea/task_472"
    # "$DATASET_BASE/pour_black_tea/task_485"
    # "$DATASET_BASE/pour_green_tea/task_486"
    # "$DATASET_BASE/pour_green_tea/task_488"
    # "$DATASET_BASE/pour_green_tea/task_498"
    # "$DATASET_BASE/pour_green_tea/task_522"
    # "$DATASET_BASE/pour_black_tea/task_523"
    # "$DATASET_BASE/pour_black_tea/task_535"
    # "$DATASET_BASE/pour_green_tea/task_536"
    # # 放置杯子到托盘上
    # "$DATASET_BASE/place_black_tea/task_425"
    # "$DATASET_BASE/place_green_tea/task_426"
    # "$DATASET_BASE/place_black_tea/task_457"
    # "$DATASET_BASE/place_green_tea/task_459"
    # "$DATASET_BASE/place_black_tea/task_501"
    # "$DATASET_BASE/place_green_tea/task_502"
    # # 取杯子到桌子上
    # "$DATASET_BASE/pick_green_tea_cup/task_513"
    # "$DATASET_BASE/pick_green_tea_cup/task_548"
    # "$DATASET_BASE/pick_black_tea_cup/task_514"
    # "$DATASET_BASE/pick_black_tea_cup/task_549"

    # # 合大倒茶异构机器人数据
    # "$DATASET_BASE/pour_black_tea/robot_632/task_571"
    # "$DATASET_BASE/pour_green_tea/robot_632/task_572"
    # "$DATASET_BASE/pour_green_tea/robot_666/task_659"
    # "$DATASET_BASE/pour_green_tea/robot_666/task_666"
    # "$DATASET_BASE/pour_black_tea/robot_666/task_715"
    # "$DATASET_BASE/pour_green_tea/robot_666/task_716"
    # "$DATASET_BASE/place_black_tea/robot_666/task_661"
    # "$DATASET_BASE/place_green_tea/robot_666/task_662"
    # "$DATASET_BASE/place_black_tea/robot_666/task_714"
    # "$DATASET_BASE/place_green_tea/robot_666/task_713"
    # "$DATASET_BASE/pick_green_tea_cup/robot_632/task_573"
    # "$DATASET_BASE/pick_black_tea_cup/robot_632/task_574"
    # "$DATASET_BASE/pick_green_tea_cup/robot_666/task_657"
    # "$DATASET_BASE/pick_black_tea_cup/robot_666/task_658"
    # "$DATASET_BASE/pick_green_tea_cup/robot_666/task_708"
    # "$DATASET_BASE/pick_black_tea_cup/robot_666/task_709"

    # # 广州数据
    # $DATASET_BASE/A2D0021B901287/task_822
    # $DATASET_BASE/A2D0021B901287/task_823
    # $DATASET_BASE/A2D0021B901287/task_824
    # $DATASET_BASE/A2D0021B901287/task_825
    # $DATASET_BASE/A2D0021B901287/task_829
    # $DATASET_BASE/A2D0021B901287/task_831
    # $DATASET_BASE/A2D0021B901287/task_835
    # $DATASET_BASE/A2D0021B901287/task_836
    # $DATASET_BASE/A2D0021B901287/task_848
    # $DATASET_BASE/A2D0021B901287/task_849


    # # 移动星厨：coffee
    # "/data/shared_datasets/star_chef/coffee/lerobot/task_867"
    # "/data/shared_datasets/star_chef/coffee/lerobot/task_868"
    # "/data/shared_datasets/star_chef/coffee/lerobot/task_869"
    # "/data/shared_datasets/star_chef/coffee/lerobot/task_872"
    # "/data/shared_datasets/star_chef/coffee/lerobot/task_873"


    # # 双手倒茶
    # "/data/shared_datasets/dual_hands/pour_tea/lerobot/task_731"
    # "/data/shared_datasets/dual_hands/pour_tea/lerobot/task_860"
    # "/data/shared_datasets/dual_hands/pour_tea/lerobot/task_874"


    # # 桌面清洁
    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_942"
    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_943"
    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_944"
    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_945"

    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_946"
    # "/data/shared_datasets/star_chef/tidy_up/lerobot/task_951"

    "/data/shared_datasets/star_chef/tidy_up/lerobot/task_964"
    "/data/shared_datasets/star_chef/tidy_up/lerobot/task_968"
)

srun torchrun \
--nnodes $SLURM_NNODES \
--nproc_per_node $(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}') \
--rdzv_id=$SLURM_JOB_ID \
--rdzv-backend=c10d \
--rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
gr00t/experiment/launch_finetune.py \
--embodiment-tag A2D \
--num-gpus 16 \
--save-total-limit 5 \
--save-steps 2000 \
--max-steps 10000 \
--global-batch-size 1024 \
--dataloader-num-workers 4 \
--base-model-path /data/shared_checkpoints/gr00t_pretrain/GR00T-N1.6-3B \
--output-dir /data/users/zhaolei/output/gr00t/ori_base/tidy_up/whole_d964_968_rel \
--dataset-path "${DATASET_PATH[@]}"

# --base-model-path /data/shared_checkpoints/gr00t_pretrain/GR00T-N1.6-3B \
# --base-model-path /data/users/zhaolei/output/gr00t/ori_base-progress/heda_pour_tea/relative-pretrain_old_robot \
