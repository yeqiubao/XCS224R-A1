#!/bin/bash
# Helper script to run BC experiments and check results

cd src/submission
source ../../.venv/bin/activate

# Set MuJoCo to use EGL for headless rendering
export MUJOCO_GL=egl

echo "=========================================="
echo "Running BC on Ant-v4 (should achieve >30% of expert)"
echo "=========================================="

python run_hw1.py \
    --expert_policy_file xcs224r/policies/experts/Ant.pkl \
    --env_name Ant-v4 \
    --exp_name bc_ant \
    --n_iter 1 \
    --expert_data xcs224r/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 5000 \
    --batch_size 10000 \
    --num_agent_train_steps_per_iter 1000

echo ""
echo "=========================================="
echo "Running BC on HalfCheetah-v4 (likely to fail <30%)"
echo "=========================================="

python run_hw1.py \
    --expert_policy_file xcs224r/policies/experts/HalfCheetah.pkl \
    --env_name HalfCheetah-v4 \
    --exp_name bc_halfcheetah \
    --n_iter 1 \
    --expert_data xcs224r/expert_data/expert_data_HalfCheetah-v4.pkl \
    --video_log_freq -1 \
    --eval_batch_size 5000 \
    --batch_size 10000 \
    --num_agent_train_steps_per_iter 1000

echo ""
echo "=========================================="
echo "Experiments complete!"
echo "Check the output above for Eval_AverageReturn values"
echo "=========================================="
