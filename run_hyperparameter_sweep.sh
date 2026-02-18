#!/bin/bash
# Script to run BC experiments with varying training steps per iteration

cd src/submission
source ../../.venv/bin/activate

# Set MuJoCo to use EGL for headless rendering
export MUJOCO_GL=egl

echo "=========================================="
echo "Running BC Hyperparameter Sweep"
echo "Varying: num_agent_train_steps_per_iter"
echo "Environment: Hopper-v4"
echo "=========================================="

# Array of training steps to test
TRAINING_STEPS=(250 500 1000 2000 4000 8000)

for steps in "${TRAINING_STEPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running with training_steps_per_iter = $steps"
    echo "=========================================="
    
    python run_hw1.py \
        --expert_policy_file xcs224r/policies/experts/Hopper.pkl \
        --env_name Hopper-v4 \
        --exp_name bc_hopper_steps_${steps} \
        --n_iter 1 \
        --expert_data xcs224r/expert_data/expert_data_Hopper-v4.pkl \
        --video_log_freq -1 \
        --eval_batch_size 5000 \
        --batch_size 10000 \
        --num_agent_train_steps_per_iter ${steps} \
        2>&1 | tee ../../hyperparameter_sweep_steps_${steps}.log
done

echo ""
echo "=========================================="
echo "Hyperparameter sweep complete!"
echo "=========================================="
