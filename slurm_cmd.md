查看详细分配情况
`groups`
#### 第一阶段：资源申请与工具初始化

Bash

```
# 1. 申请 A100 节点（账号 e32695）
srun --account=e32695 --partition=gengpu --nodes=1 --gres=gpu:a100:1 --time=02:00:00 --pty /bin/bash

# 2. 加载集群模块
module purge
module load python/anaconda3 cuda/12.1.0-gcc-11.2.0 git
```
**提示：** 如果你想用最新的 **H100**（Quest 13 节点），可以将 `--gres=gpu:a100:1` 改为 `--gres=gpu:h100:1`。
如果不指定显卡`--gres=gpu:1