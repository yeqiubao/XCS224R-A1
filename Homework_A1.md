# Homework A1 Report

## Experiment Configuration

All experiments used **identical hyperparameters** for fair comparison. These were set via command-line arguments in the experiment script:

| Hyperparameter | Value | Source/Note |
|----------------|-------|-------------|
| **Network Architecture** | 2 layers, 64 units per layer | `--n_layers 2`, `--size 64` |
| **Learning Rate** | 5e-3 | `--learning_rate 5e-3` (default) |
| **Training Steps per Iteration** | 1000 | `--num_agent_train_steps_per_iter 1000` (default) |
| **Training Batch Size** | 10,000 timesteps | `--batch_size 10000` (recommended for final results) |
| **Evaluation Batch Size** | 5,000 timesteps | `--eval_batch_size 5000` |
| **Training Batch Size (per gradient step)** | 100 | Default from `run_hw1.py` |
| **Episode Length** | 1000 steps | Environment default (`env.spec.max_episode_steps`) |
| **Number of Iterations** | 1 | `--n_iter 1` (vanilla BC) |

**Source**: These hyperparameters were specified in `run_bc_experiments.sh` and match the defaults in `run_hw1.py` for parameters not explicitly overridden.

**Note on Evaluation**: With `eval_batch_size=5000` timesteps and `ep_len=1000`, we collect multiple rollouts per evaluation. The exact number depends on episode length:
- **Ant-v4**: 5000 timesteps ÷ 1000 steps/episode = **5 rollouts**
- **Hopper-v4**: 5000 timesteps ÷ 263.05 steps/episode ≈ **19 rollouts** (episodes terminate early)

The `Eval_AverageReturn` and `Eval_StdReturn` represent the mean and standard deviation over these multiple rollouts.

---

## Results Table

| Environment | BC Mean Return | BC Std Return | Number of Rollouts | Expert Return | % of Expert | Status |
|-------------|----------------|---------------|-------------------|---------------|------------|--------|
| **Ant-v4** | 4573.85 | 135.19 | 5 | 4776.37 | **95.8%** | Pass (≥30%) |
| **Hopper-v4** | 888.49 | 99.84 | ~19 | 3717.55 | **23.9%** | Fail (<30%) |

---

## Detailed Results

### Task 1: Ant-v4 (Success Case)

**BC Policy Performance:**
- Mean Return: **4573.85**
- Standard Deviation: **135.19**
- Number of Evaluation Rollouts: **5** (from eval_batch_size=5000 timesteps ÷ 1000 steps/episode)
- Average Episode Length: 1000.0 steps (full episodes)

**Expert Policy Performance:**
- Mean Return: 4776.37
- Standard Deviation: 65.96

**Performance Comparison:**
- BC achieves **95.8%** of expert performance
- **Status**: **PASSES** the 30% threshold requirement

**Analysis**: BC works exceptionally well on Ant-v4. The policy successfully imitates the expert and maintains high performance, demonstrating that behavior cloning can be effective when distribution shift is minimal.

---

### Task 2: Hopper-v4 (Failure Case)

**BC Policy Performance:**
- Mean Return: **888.49**
- Standard Deviation: **99.84**
- Number of Evaluation Rollouts: **~19** (from eval_batch_size=5000 timesteps ÷ 263.05 steps/episode, episodes terminate early)
- Average Episode Length: 263.05 steps (episodes terminate early, vs. full 1000 steps)

**Expert Policy Performance:**
- Mean Return: 3717.55
- Standard Deviation: 3.38

**Performance Comparison:**
- BC achieves **23.9%** of expert performance
- **Status**: **FAILS** the 30% threshold requirement

**Analysis**: BC struggles significantly on Hopper-v4. The policy fails to maintain balance and episodes terminate early (average length of 263.05 steps vs. full 1000 steps). 

**Evidence for distribution shift and compounding errors:**
1. **Early termination**: Episodes end at ~263 steps instead of 1000, suggesting the policy deviates from expert trajectories
2. **Low performance**: Only 23.9% of expert performance indicates the policy makes mistakes
3. **Comparison with Ant**: Ant succeeds (95.8%) with same hyperparameters, suggesting Hopper is more sensitive to errors
4. **Standard BC failure pattern**: This pattern (early termination, compounding errors) is characteristic of distribution shift in behavior cloning literature

**Inference**: The early termination and low performance suggest the policy encounters states during evaluation that differ from those in expert demonstrations. Small initial errors compound over time, leading to states the policy hasn't seen during training, causing it to fail and terminate early.

---

## Key Observations

1. **Ant-v4 Success**: BC achieves 95.8% of expert performance, demonstrating that behavior cloning can work well when:
   - The task is relatively forgiving
   - Distribution shift is minimal
   - The policy can recover from small errors

2. **Hopper-v4 Failure**: BC achieves only 23.9% of expert performance, demonstrating the limitations of behavior cloning:
   - **Distribution shift** (inferred): The policy encounters states not seen in training, as evidenced by early termination (263.05 vs 1000 steps)
   - **Compounding errors** (inferred): Small mistakes lead to early termination, suggesting errors accumulate over time
   - **Lack of exploration**: BC only learns from expert demonstrations, not from mistakes, so it cannot recover from deviations

3. **Episode Length**: The difference in average episode length (1000.0 vs 263.05) clearly shows that the Hopper policy fails to maintain balance, while the Ant policy completes full episodes.

---

## Conclusion

This comparison demonstrates both the strengths and limitations of behavior cloning:

- **Strengths**: Can achieve near-expert performance (95.8%) when the task is forgiving and errors don't compound (Ant-v4)

- **Limitations**: Fails (23.9% performance) when small errors lead to early termination (Hopper-v4). The evidence suggests this is due to:
  - **Distribution shift**: Policy encounters states not in training data (inferred from early termination)
  - **Compounding errors**: Small mistakes accumulate, leading to failure (inferred from pattern of early termination)
  - **No error recovery**: BC only learns from expert demonstrations, not from mistakes

**Note**: "Distribution shift" and "compounding errors" are inferences based on the observed pattern (early termination, low performance) and standard behavior cloning theory. They are not directly measured but explain the observed failure mode.

The results validate the need for more advanced techniques like DAgger, which addresses the distribution shift problem by iteratively collecting data from the current policy and relabeling with expert actions.

---

## Hyperparameter Analysis

### Chosen Hyperparameter: Number of Training Steps per Iteration

**Rationale**: The number of training steps per iteration (`num_agent_train_steps_per_iter`) directly controls how much the BC agent learns from the expert data. This hyperparameter is critical because it determines the trade-off between training time and performance. Too few steps may result in underfitting, while too many steps could lead to overfitting or diminishing returns. Understanding this relationship helps optimize BC training efficiency.

### Experimental Setup

I varied `num_agent_train_steps_per_iter` from 250 to 8000 steps while keeping all other hyperparameters constant (same as the main experiments). The experiment was conducted on Hopper-v4, which showed poor performance in the main results, to see if increased training could improve performance.

### Results

| Training Steps | BC Mean Return | % of Expert |
|----------------|----------------|-------------|
| 250 | 284.89 | 7.7% |
| 500 | 741.58 | 19.9% |
| 1000 | 888.49 | 23.9% |
| 2000 | 903.49 | 24.3% |
| 4000 | 899.79 | 24.2% |
| 8000 | 1379.56 | 37.1% |

![BC Performance vs Training Steps](bc_hyperparameter_sweep.png)

### Analysis

The results show a clear relationship between training steps and performance:

1. **Rapid improvement (250-1000 steps)**: Performance increases significantly from 284.89 to 888.49 return, demonstrating that the model benefits from more training.

2. **Diminishing returns (1000-4000 steps)**: Performance plateaus around 900 return, suggesting that additional training provides minimal benefit in this range.

3. **Significant jump at 8000 steps**: Performance increases dramatically to 1379.56 return (37.1% of expert), nearly passing the 30% threshold. This suggests that very high training steps are needed for Hopper-v4, but the improvement comes at a significant computational cost.

**Key Insight**: While increasing training steps improves performance, the relationship is non-linear. For Hopper-v4, substantial training (8000 steps) is required to achieve reasonable performance, but even then, the agent only reaches 37.1% of expert performance, highlighting the fundamental limitations of BC on this task.

---

## DAgger Results

### Experimental Setup

We ran DAgger experiments on both Ant-v4 and Hopper-v4 to compare with BC performance. DAgger addresses the distribution shift problem by iteratively collecting data from the current policy and relabeling with expert actions.

**Configuration:**
- **Number of Iterations**: 10 (DAgger requires n_iter > 1)
- **Network Architecture**: 2 layers, 64 units per layer (same as BC)
- **Learning Rate**: 5e-3 (same as BC)
- **Training Steps per Iteration**: 1000 (same as BC)
- **Training Batch Size**: 10,000 timesteps per iteration (same as BC)
- **Evaluation Batch Size**: 5,000 timesteps (same as BC)
- **Episode Length**: 1000 steps (environment default)

**DAgger Process:**
1. **Iteration 0**: Train on expert data (same as BC)
2. **Iterations 1-9**: 
   - Collect trajectories using current policy
   - Relabel actions with expert policy
   - Train on combined dataset (expert + relabeled data)

### Results

![DAgger Learning Curves](dagger_learning_curves.png)

#### Ant-v4 DAgger Results

| Iteration | Mean Return | Std Return | % of Expert |
|-----------|-------------|------------|------------|
| 0 (BC) | 4573.85 | 135.19 | 95.8% |
| 1 | 4469.37 | 350.66 | 93.6% |
| 2 | 4661.96 | 63.47 | 97.6% |
| 3 | 4606.27 | 51.08 | 96.4% |
| 4 | 3795.12 | 1369.12 | 79.5% |
| 5 | 4588.24 | 99.20 | 96.1% |
| 6 | 4712.77 | 76.21 | 98.7% |
| 7 | 4802.35 | 91.42 | 100.5% |
| 8 | 4681.51 | 167.36 | 98.0% |
| 9 | 4798.53 | 102.66 | 100.5% |

**Analysis:**
- **BC Performance (Iteration 0)**: 4573.85 return (95.8% of expert)
- **Final DAgger Performance (Iteration 9)**: 4798.53 return (100.5% of expert)
- **Improvement**: DAgger achieves **expert-level performance** (100.5% of expert), slightly exceeding BC
- **Stability**: Performance is consistent across iterations, with some variance in iteration 4
- **Key Insight**: Ant-v4 already performs well with BC, so DAgger provides marginal improvement

#### Hopper-v4 DAgger Results

| Iteration | Mean Return | Std Return | % of Expert |
|-----------|-------------|------------|------------|
| 0 (BC) | 888.49 | 99.84 | 23.9% |
| 1 | 897.76 | 84.06 | 24.2% |
| 2 | 1410.96 | 293.27 | 37.9% |
| 3 | 1684.94 | 695.79 | 45.3% |
| 4 | 3307.29 | 628.07 | 89.0% |
| 5 | 2686.28 | 798.24 | 72.3% |
| 6 | 3579.01 | 190.81 | 96.3% |
| 7 | 3335.04 | 798.00 | 89.7% |
| 8 | 3711.47 | 11.33 | 99.8% |
| 9 | 3708.34 | 6.96 | 99.7% |

**Analysis:**
- **BC Performance (Iteration 0)**: 888.49 return (23.9% of expert) - **FAILS** 30% threshold
- **Final DAgger Performance (Iteration 9)**: 3708.34 return (99.7% of expert) - **PASSES** 30% threshold
- **Improvement**: DAgger achieves **near-expert performance** (99.7% of expert), a dramatic 317% improvement over BC
- **Learning Curve**: Shows clear improvement from iteration 0 to 8, with convergence near expert performance
- **Key Insight**: DAgger successfully addresses the distribution shift problem that BC failed to solve

### Comparison: BC vs DAgger

| Environment | BC Performance | DAgger Performance | Improvement |
|-------------|----------------|-------------------|-------------|
| **Ant-v4** | 4573.85 (95.8% of expert) | 4798.53 (100.5% of expert) | +4.9% (marginal) |
| **Hopper-v4** | 888.49 (23.9% of expert) | 3708.34 (99.7% of expert) | +317% (dramatic) |

### Key Observations

1. **Ant-v4**: Both BC and DAgger perform well, with DAgger achieving expert-level performance. The improvement is marginal because BC already solved the task effectively.

2. **Hopper-v4**: DAgger dramatically outperforms BC, achieving 99.7% of expert performance compared to BC's 23.9%. This demonstrates DAgger's effectiveness at addressing distribution shift.

3. **Learning Curves**: 
   - **Ant-v4**: Stable performance throughout, with slight improvement over iterations
   - **Hopper-v4**: Clear learning curve showing improvement from 23.9% to 99.7% of expert performance

4. **Distribution Shift**: DAgger successfully mitigates the distribution shift problem by:
   - Collecting data from the current policy (encountering states the policy will see)
   - Relabeling with expert actions (providing correct labels for those states)
   - Iteratively improving the policy's performance on its own trajectory distribution

### Conclusion

DAgger effectively addresses the distribution shift problem that limits BC performance, particularly on challenging tasks like Hopper-v4. While BC fails to reach 30% of expert performance on Hopper-v4, DAgger achieves near-expert performance (99.7%) through iterative data collection and expert relabeling. This validates DAgger as a superior approach for imitation learning when distribution shift is significant.
