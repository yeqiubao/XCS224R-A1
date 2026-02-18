#!/bin/bash
# Script to run DAgger experiments on Ant and Hopper

cd src/submission
source ../../.venv/bin/activate

# Set MuJoCo to use EGL for headless rendering
export MUJOCO_GL=egl

echo "=========================================="
echo "Running DAgger on Ant-v4"
echo "=========================================="

python run_hw1.py \
    --expert_policy_file xcs224r/policies/experts/Ant.pkl \
    --env_name Ant-v4 \
    --exp_name dagger_ant \
    --n_iter 10 \
    --do_dagger \
    --expert_data xcs224r/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 5000 \
    --batch_size 10000 \
    --num_agent_train_steps_per_iter 1000 \
    2>&1 | tee ../../dagger_ant.log

echo ""
echo "=========================================="
echo "Running DAgger on Hopper-v4"
echo "=========================================="

python run_hw1.py \
    --expert_policy_file xcs224r/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 \
    --exp_name dagger_hopper \
    --n_iter 10 \
    --do_dagger \
    --expert_data xcs224r/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 5000 \
    --batch_size 10000 \
    --num_agent_train_steps_per_iter 1000 \
    2>&1 | tee ../../dagger_hopper.log

echo ""
echo "=========================================="
echo "DAgger experiments complete!"
echo "=========================================="
