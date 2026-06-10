# H200 本地模型服务重启手册

本文档记录 H200 上 Signpost 实验依赖的本地服务、GPU 放置、停止/重启命令和健康检查命令。

重启命令来源：

```text
1. `ps -ww -fp <pid>` 读取当前正在运行服务的完整 CMD。
2. `/proc/<pid>/environ` 读取当前正在运行服务的 `CUDA_VISIBLE_DEVICES` 和 conda 环境。
3. `tmux capture-pane` 只用于确认服务日志和 tmux session，不以人工猜测替代当前运行命令。
```

因此下面的 vLLM 命令是按当前实际运行进程复原，不是重新设计的新命令。

记录时间：2026-06-03 21:48 CST 左右。

## 1. 当前服务与 GPU 放置

| 端口 | 服务 | 模型 | 当前 GPU | 主进程 | Engine 进程 | 显存 |
| --- | --- | --- | --- | --- | --- | --- |
| 8000 | chat | `/data/srl/Llama-3.3-70B-FP8` | GPU 1 | `3589065` | `3589990` | 约 139.6 GiB |
| 8001 | embedding | `/data/srl/nemotron-8b` | GPU 2 | `3284259` | `3289877` | 约 31.5 GiB |
| 8002 | GraphRAG-R1 chat | `/data/srl/GraphRAG-R1` served by Qwen2.5 + LoRA | GPU 0 | `2781536` | `2782118` | 约 111 GiB |
| 8003 | HiPRAG chat | `/data/srl/HiPRAG-7B` | GPU 2 | `3171979` | `3172384` | 约 108.5 GiB |
| 8033 | rerank | `/data/srl/llama-nemotron-rerank-1b-v2` | GPU 0 | `2762682` | `2766293` | 约 25.4 GiB |
| 9200 | Elasticsearch | `/data/srl/elasticsearch-8.12.1-signpost` | CPU | `1332585` | - | - |

当前 GPU 占用结论：

```text
GPU 0: 8002 GraphRAG-R1 + 8033 rerank
GPU 1: 8000 Llama-3.3-70B-FP8
GPU 2: 8001 embedding + 8003 HiPRAG
```

没有天然空闲卡。若停掉 HiPRAG，可释放 GPU 2 上约 108.5 GiB，但 GPU 2 仍保留 embedding 服务。

## 2. 停服务前检查

先确认是否有实验还会用对应服务：

```bash
ps -eo pid,ppid,user,stat,lstart,cmd | rg 'run_signpost|run_baseline|signpost.agent|legal_q100|hiprag|graphrag_r1|cluerag|musique'
tmux ls
```

当前观察到的风险：

```text
formal-legal-q100 的无人值守命令还在运行。
该命令后续包含:
  scripts/baselines/run_baseline_method.sh graphrag_r1 ...
  scripts/baselines/run_baseline_method.sh hiprag ...

因此如果现在停 8003 HiPRAG，legal_q100 后续跑到 hiprag 阶段前必须先重启 8003。
否则该阶段会失败。
```

## 3. 通用环境

所有 vLLM 服务使用同一个环境：

```bash
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
```

建议在对应 tmux 中启动服务，便于保留日志。若原 session 已存在，使用 `tmux attach -t <session-name>` 回到原窗口启动；只有 session 不存在时才使用 `tmux new -s <session-name>`。

```bash
tmux new -s <session-name>
```

本机存在代理环境变量时，健康检查必须绕过代理：

```bash
curl --noproxy '*' ...
```

## 4. 8000 Chat 服务

用途：

```text
Signpost 主 LLM、semantic extraction、agent final generation、baseline final generation。
```

当前不建议停。GPU 1 当前高负载，说明仍在被实验使用。

重启命令。CMD 来自当前运行进程 `pid=3589065`，GPU 绑定来自 `/proc/3589065/environ`：

```bash
tmux attach -t llama
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=1
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/Llama-3.3-70B-FP8 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8000/v1/models | head
```

## 5. 8001 Embedding 服务

用途：

```text
Signpost chunk index、graph sync、baseline retrieval/indexing 共享 embedding 服务。
```

当前不建议停。embedding 是共享服务，MuSiQue 和其他数据集离线/在线流程可能继续使用。

重启命令。CMD 来自当前运行进程 `pid=3284259`，GPU 绑定来自 `/proc/3284259/environ`：

```bash
tmux attach -t embed
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=2
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/nemotron-8b \
  --runner pooling \
  --port 8001 \
  --trust-remote-code
```

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8001/v1/models | head
curl --noproxy '*' -fsS http://127.0.0.1:8001/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/nemotron-8b","input":["embedding health check"]}' | head
```

## 6. 8002 GraphRAG-R1 服务

用途：

```text
GraphRAG-R1 baseline 专用 chat 服务。
```

如果当前没有 GraphRAG-R1 baseline 运行，可以停；但 `formal-legal-q100` 的无人值守命令后续仍包含 GraphRAG-R1 阶段，停后必须在该阶段前重启。

重启命令。CMD 来自当前运行进程 `pid=2781536`，GPU 绑定来自 `/proc/2781536/environ`：

```bash
tmux attach -t gr1-vllm
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/Qwen2.5-7B-Instruct \
  --served-model-name /data/srl/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules /data/srl/GraphRAG-R1=/data/srl/GraphRAG-R1-lora/checkpoints/qwen_base_v1 \
  --port 8002 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75
```

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8002/v1/models | head
```

## 7. 8003 HiPRAG 服务

用途：

```text
HiPRAG baseline 专用 chat 服务。
```

当前 HiPRAG vLLM 服务本身没有持续请求，最近日志显示已空闲；但 `formal-legal-q100` 的无人值守命令后续仍包含 HiPRAG 阶段。

判断：

```text
如果只是想短时间释放 GPU 2 显存，可以停 8003。
但必须在 legal_q100 或其他数据集跑到 hiprag baseline 前重启。
否则对应 HiPRAG baseline 会失败。
```

重启命令。CMD 来自当前运行进程 `pid=3171979`，GPU 绑定来自 `/proc/3171979/environ`：

```bash
tmux attach -t hiprag-vllm
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=2
export VLLM_USE_DEEP_GEMM=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/HiPRAG-7B \
  --served-model-name /data/srl/HiPRAG-7B \
  --port 8003 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75
```

2026-06-04 在 H200 重新启动 HiPRAG 时，原命令会在 DeepGEMM warmup 阶段报错：
`DeepGEMM backend is not available or outdated`。本地 vLLM 代码明确支持通过
`VLLM_USE_DEEP_GEMM=0` 全局关闭 DeepGEMM，因此重启命令中保留原始 vLLM 参数，只增加该环境变量。

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8003/v1/models | head
```

停止方式：

```bash
kill -TERM 3171979
sleep 10
nvidia-smi
```

如果仍未退出，再确认没有请求后使用：

```bash
kill -KILL 3171979
```

不要直接杀 engine 子进程；优先停主进程。

## 8. 8033 Rerank 服务

用途：

```text
Signpost rerank、ClueRAG rerank、部分 hybrid retrieval/baseline rerank。
```

重启命令。CMD 来自当前运行进程 `pid=2762682`，GPU 绑定来自 `/proc/2762682/environ`：

```bash
tmux attach -t rerank
cd /data/srl/signpost_re
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/llama-nemotron-rerank-1b-v2 \
  --port 8033 \
  --trust-remote-code \
  --gpu-memory-utilization 0.3
```

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8033/v1/models | head
curl --noproxy '*' -fsS http://127.0.0.1:8033/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{"model":"/data/srl/llama-nemotron-rerank-1b-v2","query":"health check","documents":["health check document"]}' | head
```

注意：

```text
不要用未设置 --noproxy 的 curl 判断 8033 是否坏。
当前环境里代理变量可能把 localhost 请求送到代理，导致假 502。
```

## 9. 9200 Elasticsearch

用途：

```text
Signpost graph/chunk index、baseline ES index。
```

一般不建议在实验中途重启 ES。

重启命令：

```bash
tmux attach -t es
cd /data/srl/elasticsearch-8.12.1-signpost
./bin/elasticsearch
```

健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:9200 | jq .
curl --noproxy '*' -fsS http://127.0.0.1:9200/_cat/indices?v | head
```

## 10. 停 HiPRAG 后如何恢复当前实验兼容性

如果释放 GPU 2 时停了 HiPRAG，后续恢复只需要保证以下接口仍可用，实验命令不用改：

```text
HIPRAG_API_BASE=http://127.0.0.1:8003/v1
HIPRAG_CHAT_MODEL=/data/srl/HiPRAG-7B
```

也就是按第 7 节重启 8003，端口和 served model 保持不变即可。

## 11. 快速状态命令

查看端口：

```bash
ss -ltnp | rg '8000|8001|8002|8003|8033|9200'
```

查看 GPU：

```bash
nvidia-smi
nvidia-smi --query-compute-apps=pid,gpu_uuid,process_name,used_memory --format=csv,noheader,nounits
```

查看实验：

```bash
ps -eo pid,ppid,user,stat,lstart,cmd | rg 'run_signpost|run_baseline|signpost.agent|semantic_graph|legal_q100|musique'
tmux ls
```

## 12. 当前建议

如果目标是“尽快空出 GPU 2 的大块显存”，可以停 `8003 HiPRAG`，会释放约 108.5 GiB。这样不会改变任何实验设计，因为后续仍可用同端口、同模型重启。

但当前 `formal-legal-q100` 仍在运行，且命令后续包含 `hiprag` baseline。因此停掉后需要人工保证：

```text
在 legal_q100 执行到 hiprag baseline 前，重启 8003。
```

如果没人值守，不建议现在停。

## 13. GPU0 上切换 GraphRAG-R1 与 HiPRAG

适用场景：

```text
GPU2 因 embedding 或其他用户任务导致剩余显存不足以启动 HiPRAG；
GPU0 上已有 GraphRAG-R1 和 rerank；
当前暂时不需要 GraphRAG-R1，但需要启动 HiPRAG。
```

当前经验显存：

```text
GPU0 rerank: 约 25.4 GiB
GPU0 GraphRAG-R1: 约 111 GiB
GPU0 停 GraphRAG-R1 后剩余: 约 117 GiB
HiPRAG: 约 108.5 GiB
```

因此可以在 GPU0 保留 rerank，暂停 GraphRAG-R1，然后在 GPU0 启动 HiPRAG。对实验代码无影响，只要端口和模型名保持：

```text
GraphRAG-R1: http://127.0.0.1:8002/v1, model /data/srl/GraphRAG-R1
HiPRAG: http://127.0.0.1:8003/v1, model /data/srl/HiPRAG-7B
```

切换前检查 GraphRAG-R1 没有活动请求：

```bash
ss -tnp | rg ':8002|State'
ps -eo pid,ppid,user,stat,lstart,cmd | rg 'run_baseline|graphrag_r1|8002'
```

停 GraphRAG-R1：

```bash
kill -TERM 2781536
sleep 15
ss -ltnp | rg ':8002|State'
nvidia-smi
```

如果 `8002` 仍在监听，再确认没有请求后：

```bash
kill -KILL 2781536
sleep 5
nvidia-smi
```

在原 HiPRAG tmux 窗口里启动 HiPRAG 到 GPU0：

```bash
tmux attach -t hiprag-vllm
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=0
export VLLM_USE_DEEP_GEMM=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/HiPRAG-7B \
  --served-model-name /data/srl/HiPRAG-7B \
  --port 8003 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75
```

2026-06-04 实测：在 `hiprag-vllm` 窗口里使用上面命令已成功启动到 GPU0，`8003` 正常监听，
`/v1/models` 返回 served model `/data/srl/HiPRAG-7B`。不加 `VLLM_USE_DEEP_GEMM=0` 时会在
DeepGEMM warmup 失败。

HiPRAG 健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8003/v1/models | head
```

需要恢复 GraphRAG-R1 时，先停 HiPRAG：

```bash
ss -ltnp | rg ':8003|State'
ps -eo pid,ppid,user,stat,lstart,cmd | rg 'HiPRAG-7B|8003'
kill -TERM <hiprag-main-pid>
sleep 15
nvidia-smi
```

然后在原 GraphRAG-R1 tmux 窗口里按当前运行进程记录的原命令重启：

```bash
tmux attach -t gr1-vllm
cd /data/srl
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate /data/srl/.conda_envs/vllm
export CUDA_VISIBLE_DEVICES=0
python -m vllm.entrypoints.openai.api_server \
  --model /data/srl/Qwen2.5-7B-Instruct \
  --served-model-name /data/srl/Qwen2.5-7B-Instruct \
  --enable-lora \
  --lora-modules /data/srl/GraphRAG-R1=/data/srl/GraphRAG-R1-lora/checkpoints/qwen_base_v1 \
  --port 8002 \
  --trust-remote-code \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.75
```

GraphRAG-R1 健康检查：

```bash
curl --noproxy '*' -fsS http://127.0.0.1:8002/v1/models | head
```
