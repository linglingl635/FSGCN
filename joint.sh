#!/bin/bash
#SBATCH -o gpu.%j.out ##作业的输出信息文件
#SBATCH -J joint ##作业名
#SBATCH -A pi_zhanglingyan ##指定计费账户
#SBATCH -p gpu8Q ##使用gpu2Q计算队列  *
#SBATCH -q gpuq ## qos服务，作业提交到cpuQ分区以外需要指定 **
#SBATCH --nodes=1 ##申请1个节点
#SBATCH --ntasks-per-node=1 ##每节点运行1个任务（进程）
#SBATCH --gres=gpu:1 ##每个作业占用的GPU数量 *

python main.py --config config/nturgbd-cross-view/fsgcn.yaml

