# Retool RL Training Script - Claude Guide

This directory contains the retool example for training language models with tool-enabled reinforcement learning (RL).

## Script Overview: `retool_qwen3_4b_rl.sh`

This script trains a Qwen3-4B model using GRPO (Group Relative Policy Optimization) with tool support for mathematical reasoning tasks. The model learns to use a Python code interpreter to solve complex math problems through trial and error.

---

## How It Works

### Prompt Assembly

The system uses a **Jinja2 template** (`TOOL_TEMPLATE` in `generate_with_retool.py`) to structure conversations:

#### Initial Prompt Structure:
```
<|im_start|>system
You are a helpful assistant that can use Python tools to solve mathematical problems.

# Tools

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "code_interpreter", ...}}
</tools>

For each function call, return a json object with function name and arguments:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
<|im_end|>
<|im_start|>user
[Math problem]<|im_end|>
<|im_start|>assistant
```

#### Multi-Turn Agent Loop:
The `generate()` function implements an iterative reasoning loop (up to 16 turns):

1. **Model generates** → Response with text or `<tool_call>`
2. **Extract action** → Parse for tool calls, code blocks, or final answer
3. **Execute tool** → Run code in sandbox if detected
4. **Append observation** → Add `<interpreter>\n[result]\n</interpreter>` to context
5. **Continue** → Model sees its response + tool output, generates next action
6. **Repeat** → Until final answer `Answer: \boxed{...}` or max turns reached

**Example conversation flow:**
```
Turn 1: "Let me solve this step by step.
         <tool_call>{"name": "code_interpreter", "arguments": {"code": "x = 25 * 4\nprint(x)"}}</tool_call>"
         → System adds: "\n\n<interpreter>\nOutput:\n100\n</interpreter>\n\n"

Turn 2: "Now I'll use that result.
         <tool_call>{"name": "code_interpreter", "arguments": {"code": "result = 100 + 50\nprint(result)"}}</tool_call>"
         → System adds: "\n\n<interpreter>\nOutput:\n150\n</interpreter>\n\n"

Turn 3: "Answer: \boxed{150}"
         → Done! (detected final answer)
```

**Key features:**
- The cumulative `prompt + response` grows with each turn
- Loss masks: Model's text has `loss_mask=1`, tool outputs have `loss_mask=0`
- Only the model's generated tokens contribute to training loss
- Tool call count tracked for reward calculation

---

### Sandbox Environment

The **`PythonSandbox`** class provides secure code execution with multiple safety layers:

#### Safety Layers:

**1. Static Code Analysis** (`_check_code_safety()`):
- Regex pattern matching to block dangerous operations
- Blocks: `os`, `sys`, `subprocess`, `eval()`, `exec()`, `open()`, `__import__`, dunder methods
- **Whitelist approach**: Only allows safe modules (`math`, `random`, `datetime`, `statistics`, `decimal`, `fractions`, `itertools`, `functools`, `operator`, `collections`)

**2. Process Isolation**:
```python
subprocess.Popen(
    ["python3", script_path],  # Separate Python process
    stdout=PIPE, stderr=PIPE,   # Captured I/O
    env=isolated_env,           # Clean environment
    cwd=temp_directory,         # Isolated filesystem
    timeout=120                 # 2-minute limit
)
```

**3. Resource Limits**:
- **Memory**: 4GB per process (enforced via `resource.setrlimit()`)
- **Timeout**: 120 seconds max execution time per tool call
- **Concurrency**: Semaphore limits 32 concurrent executions
- **Global memory**: 12GB total limit with progressive cleanup

**4. Temporary Execution Environment**:
- Creates `/tmp/python_sandbox_<random>/` for each execution
- Code written to `code.py` in isolated temp directory
- Automatic cleanup after execution (even on errors)
- No persistent filesystem access

**5. Output Capture**:
```python
# User code is wrapped:
sys.stdout = StringIO()  # Redirect output
sys.stderr = StringIO()

try:
    [user code]
    return captured_output
except Exception as e:
    return formatted_error
```

**6. Memory Management**:
Progressive cleanup based on usage:
- **> 3GB**: Light cleanup (`gc.collect()`)
- **> 6GB**: Normal cleanup
- **> 9GB**: Aggressive cleanup (multiple GC cycles, clear module caches)
- **> 12GB**: Refuse execution until memory freed

#### What the Sandbox Prevents:
- ✅ File system access (except temp directory)
- ✅ Network operations
- ✅ System calls
- ✅ Import of dangerous modules
- ✅ Infinite loops (timeout)
- ✅ Memory bombs (4GB limit)
- ✅ Process fork bombs (subprocess isolation)

---

## Key Configuration Sections

### 1. Model Checkpoints (`CKPT_ARGS`)
```bash
--hf-checkpoint /root/data/font-info/qwen3-4b-sft           # Base model for inference
--ref-load /root/data/font-info/qwen3-4b-sft_torch_dist    # Reference for KL divergence
--save /root/data/font-info/qwen3-4b-sft/qwen3-4b-sft-multi-turn/  # Save location
--save-interval 20                                          # Checkpoint every 20 iterations
--rotary-base 5000000                                       # Extended context (5M tokens)
```

### 2. Rollout Configuration (`ROLLOUT_ARGS`)
```bash
--prompt-data /root/data/dapo-math-17k/dapo-math-17k.jsonl  # Training dataset
--num-rollout 3000                                          # Total training samples
--rollout-batch-size 32                                     # Prompts per batch
--n-samples-per-prompt 8                                    # Diverse responses per prompt
--rollout-max-response-len 8192                             # Max generation length
--rollout-temperature 1                                     # Sampling temperature
--global-batch-size 256                                     # Total batch (32 × 8)
--balance-data                                              # Balance positive/negative samples
```

### 3. Evaluation (`EVAL_ARGS`)
```bash
--eval-interval 20                                          # Evaluate every 20 iterations
--eval-prompt-data aime /root/data/aime-2024/aime-2024.jsonl  # AIME 2024 benchmark
--n-samples-per-eval-prompt 16                              # Samples per eval prompt
--eval-max-response-len 16384                               # Longer for complex reasoning
```

### 4. Performance Optimization (`PERF_ARGS`)
```bash
--tensor-model-parallel-size 2                              # 2-way tensor parallelism
--sequence-parallel                                         # Enable sequence parallelism
--use-dynamic-batch-size                                    # Adjust batch by sequence length
--max-tokens-per-gpu 9216                                   # Token limit per GPU
--recompute-granularity full                                # Full activation recomputation
--recompute-num-layers 1                                    # Recompute 1 layer for memory
```

### 5. GRPO Algorithm (`GRPO_ARGS`)
```bash
--advantage-estimator grpo                                  # Group Relative Policy Optimization
--use-kl-loss                                               # Enable KL divergence constraint
--kl-loss-coef 0.00                                         # KL coefficient (can tune)
--entropy-coef 0.00                                         # Entropy bonus (can tune)
--eps-clip 0.2                                              # Lower clip bound
--eps-clip-high 0.28                                        # Upper clip bound
```

### 6. Optimizer (`OPTIMIZER_ARGS`)
```bash
--optimizer adam                                            # Adam optimizer
--lr 1e-6                                                   # Conservative learning rate for RL
--lr-decay-style constant                                   # No decay
--weight-decay 0.1                                          # L2 regularization
--adam-beta1 0.9                                            # First moment decay
--adam-beta2 0.98                                           # Second moment decay
```

### 7. Retool Integration (`CUSTOM_ARGS`)
```bash
--custom-generate-function-path generate_with_retool.generate    # Tool-enabled generation
--custom-rm-path generate_with_retool.reward_func                # Reward function
```

### 8. Reward Function

The `reward_func()` in `generate_with_retool.py` calculates rewards:

```python
# 1. Check if final answer matches ground truth (using math_dapo_compute_score)
score = compare_boxed_answer(model_answer, ground_truth)

# 2. If answer is wrong, give partial credit for tool usage
if score < 0:
    tool_call_reward = (num_turns - 2) / 2 * 0.1
    score = min(-0.6, score + tool_call_reward)  # Cap at -0.6

# This encourages the model to use tools even when final answer is wrong
```

**Reward structure:**
- ✅ Correct answer: `+1.0`
- ❌ Wrong answer, 0 tool calls: `-1.0`
- ❌ Wrong answer, 3 tool calls: `-1.0 + 0.05 = -0.95`
- ❌ Wrong answer, 10 tool calls: `-1.0 + 0.4 = -0.6` (capped)

This encourages exploration and tool usage during training.

---

## How to Modify

### Changing the Model
```bash
# Update in CKPT_ARGS:
--hf-checkpoint /path/to/your/model
--ref-load /path/to/your/model_torch_dist
--save /path/to/save/checkpoints
```

### Adjusting Training Scale
```bash
# In ROLLOUT_ARGS:
--num-rollout 5000              # More training samples
--rollout-batch-size 64         # Larger batches (if memory allows)
--n-samples-per-prompt 16       # More diverse samples

# In PERF_ARGS:
--tensor-model-parallel-size 4  # Use more GPUs
--max-tokens-per-gpu 4096       # Reduce if OOM
```

### Tuning the Algorithm
```bash
# In GRPO_ARGS:
--kl-loss-coef 0.01            # Stay closer to reference model
--entropy-coef 0.01            # Encourage more exploration
--eps-clip 0.15                # Smaller = more conservative updates

# In OPTIMIZER_ARGS:
--lr 5e-7                      # Decrease for stability
--lr 2e-6                      # Increase for faster learning
```

### Modifying Tool Behavior

Edit `TOOL_CONFIGS` in `tool_sandbox.py`:
```python
"max_turns": 16,               # Max conversation turns
"max_tool_calls": 16,          # Max tool executions
"tool_concurrency": 32,        # Concurrent tool executions
"python_timeout": 120,         # Timeout per execution (seconds)
"python_memory_limit": "4GB",  # Memory per process
```

Add new allowed modules to `PythonSandbox`:
```python
self.allowed_modules = {
    "math", "random", "datetime",
    "numpy",    # Add for numerical computation
    "sympy",    # Add for symbolic math
}
```

### Using Different Datasets
```bash
# In ROLLOUT_ARGS:
--prompt-data /path/to/your/train.jsonl
--input-key your_prompt_key     # JSON key for prompts
--label-key your_label_key      # JSON key for labels

# In EVAL_ARGS:
--eval-prompt-data my_eval /path/to/eval.jsonl
```

### GPU Configuration
```bash
# Adjust Ray cluster:
ray start --num-gpus 8          # Total GPUs available

# In submission:
--actor-num-gpus-per-node 8     # GPUs for training

# In PERF_ARGS:
--tensor-model-parallel-size 4  # Distribute model across GPUs
```

---

## Prerequisites

### 1. Model Conversion
Convert HuggingFace checkpoint to torch dist format:
```bash
source scripts/models/qwen3-4B.sh
PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/data/font-info/qwen3-4b-sft \
    --rotary-base 5000000 \
    --save /root/data/font-info/qwen3-4b-sft_torch_dist
```

### 2. Datasets
Download required datasets:
```bash
# Training data
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/data/dapo-math-17k

# Evaluation data
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/data/aime-2024

# Pre-trained SFT model (optional, to skip SFT phase)
hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /root/data/font-info/qwen3-4b-sft
```

### 3. Environment
```bash
cd miles
pip install -e . --no-deps
pip install jinja2  # Required for prompt templating
```

### 4. WANDB
Set environment variable for logging:
```bash
export WANDB_KEY=your_wandb_api_key
```

---

## Related Files

- **`generate_with_retool.py`**: Custom generation with tool support, multi-turn loop, reward calculation
- **`tool_sandbox.py`**: Safe Python code execution environment with security layers
- **`README.md`**: Setup instructions and dataset download commands
- **`sft_data_processing.py`**: Process SFT dataset for initial supervised training
- **`/root/data/miles/scripts/models/qwen3-4B.sh`**: Model architecture configuration (MODEL_ARGS)
- **`train.py`**: Main training script (in miles root directory)

---

## Common Issues

### OOM (Out of Memory)
**Symptoms**: CUDA OOM errors, process killed
**Solutions**:
- Reduce `--max-tokens-per-gpu 4096` in PERF_ARGS
- Decrease `--rollout-batch-size 16` in ROLLOUT_ARGS
- Increase recomputation: `--recompute-num-layers 2`
- Reduce `--n-samples-per-prompt 4`

### Slow Training
**Symptoms**: Low throughput, long iteration times
**Solutions**:
- Increase `--rollout-batch-size 64` if memory allows
- Reduce `--n-samples-per-prompt 4` for faster rollouts
- Check NVLink detection: Should see "HAS_NVLINK: 1" in logs
- Reduce `--eval-interval 50` to evaluate less frequently

### Ray Connection Issues
**Symptoms**: "Cannot connect to Ray cluster"
**Solutions**:
- Verify `MASTER_ADDR` environment variable is set correctly
- Check port 8265 is available: `netstat -tulpn | grep 8265`
- Kill existing Ray processes: `ray stop --force && pkill -9 ray`
- Check Ray dashboard: `http://localhost:8265`

### Poor Convergence
**Symptoms**: Reward not improving, unstable training
**Solutions**:
- Increase `--kl-loss-coef 0.01` to constrain policy updates
- Decrease learning rate: `--lr 5e-7`
- Adjust clipping: `--eps-clip 0.15 --eps-clip-high 0.2`
- Check eval metrics in WANDB for overfitting
- Ensure reference model matches initial policy

### Tool Execution Errors
**Symptoms**: "Error: Code contains dangerous pattern"
**Solutions**:
- Model trying to import blocked modules
- Add allowed modules to `PythonSandbox.allowed_modules` if safe
- Check `--python-timeout` if legitimate code timing out
- Review sandbox logs for specific pattern matches

### Memory Leaks
**Symptoms**: Memory usage grows over time
**Solutions**:
- Reduce `tool_concurrency` in TOOL_CONFIGS
- Lower `max_memory_usage` threshold for earlier cleanup
- Increase `--save-interval` to checkpoint and restart more often
- Monitor with: `watch -n 1 free -h`

---

## Monitoring

### WANDB Dashboard
- **Project**: `miles-dapo`
- **Group**: `qwen3-4B-test-multi-turn`
- **Key metrics**:
  - `reward/mean`: Average reward per rollout
  - `eval/aime/score`: Accuracy on AIME benchmark
  - `debug/tools_used`: Tool usage statistics
  - `debug/turn`: Average turns per sample

### Ray Dashboard
- **URL**: `http://localhost:8265`
- **Monitor**:
  - GPU utilization
  - Memory usage per actor
  - Task execution times
  - SGLang engine status

### Checkpoints
- Saved every 20 iterations at `--save` location
- Contains model weights, optimizer state, training step
- Can resume from checkpoint if training interrupted

---

## Execution Flow

```
1. Script starts
   ↓
2. Kill existing processes (sglang, ray, python)
   ↓
3. Detect NVLink availability
   ↓
4. Start Ray cluster (4 GPUs)
   ↓
5. Submit Ray job with runtime environment
   ↓
6. Initialize training:
   - Load model from HF checkpoint
   - Load reference model from torch dist
   - Initialize optimizer
   - Start SGLang inference engines
   ↓
7. Training loop (each iteration):
   a. Sample prompts from dataset
   b. Generate responses with tools (multi-turn)
   c. Calculate rewards
   d. Compute advantages (GRPO)
   e. Update policy with PPO
   f. Log metrics to WANDB
   g. Save checkpoint (every 20 iters)
   h. Evaluate on AIME (every 20 iters)
   ↓
8. Training complete
```

---

## Advanced Tips

### Debugging Generation
Add logging in `generate_with_retool.py`:
```python
print(f"Turn {turn}: {cur_response}")
print(f"Action: {action}, Content: {content}")
print(f"Tool output: {next_obs}")
```

### Custom Reward Shaping
Modify `reward_func()` to:
- Penalize long responses: `score -= len(response) * 0.0001`
- Reward intermediate steps: `score += 0.1 * num_correct_steps`
- Penalize syntax errors: `score -= 0.2 * num_errors`

### Multi-Turn Analysis
Track tool usage patterns:
```python
sample.tool_sequence = []  # List of tools called
sample.turn_count = turn   # Total conversation turns
```

### Batch Size Tuning
For optimal throughput:
- Start with small batch size
- Monitor GPU memory with `nvidia-smi`
- Increase batch size until ~90% GPU memory used
- Use `--use-dynamic-batch-size` to handle variable lengths

---

## Notes

- The script uses `set -ex` for verbose error reporting (print each command, exit on error)
- All processes forcefully killed at startup to ensure clean state
- `PYTHONPATH` includes: Megatron-LM, script directory, Miles root
- `CUDA_DEVICE_MAX_CONNECTIONS=1` set for compatibility with certain CUDA operations
- `NCCL_NVLS_ENABLE` dynamically set based on NVLink detection
- The reference model (`--ref-load`) is loaded once and kept frozen for KL divergence calculations
- Loss masks ensure tool outputs don't contribute to policy gradient
- Tool call count tracked separately from conversation turn count
