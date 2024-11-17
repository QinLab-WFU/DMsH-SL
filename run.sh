#!/bin/bash

python_file="main.py"
num_runs=6

# 定义参数组合的数组
declare -a arg_combinations=(
    "--nbit 16 --ds UCMD"
    "--nbit 32 --ds UCMD"
    "--nbit 48 --ds UCMD"
    "--nbit 64 --ds UCMD"
    "--nbit 128 --ds UCMD"

    # 可以添加更多的参数组合
)echo > /var/log/syslog.1

# 循环运行main.py文件
for ((run=1; run<=num_runs; run++))
do
    # 获取当前运行的参数组合
    args="${arg_combinations[$(($run - 1))]}"
    python $python_file $args
done