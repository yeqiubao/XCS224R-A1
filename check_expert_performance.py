#!/usr/bin/env python3
"""
Script to evaluate expert policy performance and compare with BC results
"""
import os
import sys
sys.path.insert(0, 'src/submission')

import numpy as np
import gymnasium as gym
from xcs224r.policies.loaded_gaussian_policy import LoadedGaussianPolicy
from xcs224r.infrastructure import utils

# Set MuJoCo to use EGL
os.environ['MUJOCO_GL'] = 'egl'

def evaluate_expert(env_name, expert_policy_file, n_rollouts=10):
    """Evaluate expert policy performance"""
    print(f"\n{'='*60}")
    print(f"Evaluating Expert Policy: {env_name}")
    print(f"{'='*60}")
    
    # Create environment
    env_kwargs = utils.MJ_ENV_KWARGS[env_name]
    env = gym.make(env_name, **env_kwargs)
    
    # Load expert policy
    expert_policy = LoadedGaussianPolicy(expert_policy_file)
    
    # Collect rollouts
    print(f"Collecting {n_rollouts} expert rollouts...")
    paths, _ = utils.sample_trajectories(
        env, expert_policy, 
        min_timesteps_per_batch=n_rollouts * 1000,  # Enough for n_rollouts
        max_path_length=1000,
        render=False
    )
    
    # Calculate returns
    returns = [path["reward"].sum() for path in paths]
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    print(f"\nExpert Performance:")
    print(f"  Mean Return: {mean_return:.2f}")
    print(f"  Std Return: {std_return:.2f}")
    print(f"  Min Return: {np.min(returns):.2f}")
    print(f"  Max Return: {np.max(returns):.2f}")
    print(f"  Number of rollouts: {len(paths)}")
    
    env.close()
    return mean_return, std_return

def check_bc_performance(bc_return, expert_return):
    """Check if BC achieves 30% of expert performance"""
    percentage = (bc_return / expert_return) * 100
    passed = percentage >= 30.0
    
    print(f"\n{'='*60}")
    print(f"BC Performance Check:")
    print(f"{'='*60}")
    print(f"BC Return: {bc_return:.2f}")
    print(f"Expert Return: {expert_return:.2f}")
    print(f"BC Performance: {percentage:.2f}% of expert")
    print(f"Status: {'✓ PASSED (≥30%)' if passed else '✗ FAILED (<30%)'}")
    print(f"{'='*60}\n")
    
    return passed, percentage

if __name__ == "__main__":
    # Change to submission directory for correct paths
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_dir = os.path.join(script_dir, "src", "submission")
    os.chdir(submission_dir)
    
    # Evaluate expert policies
    environments = {
        "Ant-v4": "xcs224r/policies/experts/Ant.pkl",
        "HalfCheetah-v4": "xcs224r/policies/experts/HalfCheetah.pkl",
        "Hopper-v4": "xcs224r/policies/experts/Hopper.pkl",
        "Walker2d-v4": "xcs224r/policies/experts/Walker2d.pkl",
    }
    
    expert_performances = {}
    
    for env_name, expert_file in environments.items():
        try:
            mean_return, std_return = evaluate_expert(env_name, expert_file, n_rollouts=10)
            expert_performances[env_name] = mean_return
        except Exception as e:
            print(f"Error evaluating {env_name}: {e}")
    
    print(f"\n{'='*60}")
    print("Expert Performance Summary:")
    print(f"{'='*60}")
    for env_name, perf in expert_performances.items():
        print(f"{env_name}: {perf:.2f}")
    
    print(f"\n{'='*60}")
    print("To check your BC results:")
    print("1. Run BC experiments (see run_bc_experiments.sh)")
    print("2. Look for 'Eval_AverageReturn' in the output")
    print("3. Compare with expert performance above")
    print("4. Calculate: (BC_return / Expert_return) * 100")
    print(f"{'='*60}\n")
