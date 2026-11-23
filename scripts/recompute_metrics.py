# -*- coding: utf-8 -*-
"""
不跑模型，直接从两份文件重算 L2 指标。

用途：验证自己对指标算法的理解对不对。如果手算结果和日志里的表格对得上，
说明读懂源码了；对不上就说明哪里理解错了。

需要的两份文件：
    - nuscenes_infos_val.pkl    GT（标准答案），从 data/infos/mini/ 拿
    - results_*.pkl             答卷（预测），评测脚本输出

用法：
    python recompute_metrics.py <infos_val.pkl> <results.pkl>
"""
import sys
import pickle
import numpy as np


def load_results(path):
    """读答卷。文件里存的是 CUDA tensor，没 GPU 的机器要先打补丁，
    否则报 Attempting to deserialize object on a CUDA device。"""
    import torch.serialization as S
    S.default_restore_location = lambda storage, location: storage.cpu()
    with open(path, "rb") as f:
        return pickle.load(f)


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def main(infos_path, results_path):
    with open(infos_path, "rb") as f:
        infos = pickle.load(f)["infos"]
    res = load_results(results_path)

    print(f"GT 样本数: {len(infos)}   预测样本数: {len(res)}")

    l2_list = []
    skipped = 0
    for i in range(len(infos)):
        mask = to_np(infos[i]["gt_ego_fut_masks"]).astype(bool)
        # 掩码不全的样本评测代码会跳过，这里跟着跳，否则分母对不上
        if not mask.all():
            skipped += 1
            continue
        # GT 存的是增量，累加成绝对位置
        gt = np.cumsum(to_np(infos[i]["gt_ego_fut_trajs"]), axis=0)
        # 预测直接就是绝对位置
        pred = to_np(res[i]["img_bbox"]["final_planning"])
        l2_list.append(np.sqrt(((pred - gt) ** 2).sum(axis=-1)))

    l2 = np.array(l2_list)
    print(f"参与统计: {len(l2)}   跳过: {skipped}")

    # 每个时刻的瞬时 L2
    per_t = l2.mean(axis=0)
    # 表格里显示的是累积平均
    cumavg = np.array([per_t[:k + 1].mean() for k in range(len(per_t))])
    # avg 只取 1.0s / 2.0s / 3.0s 三列（索引 1、3、5）
    avg = np.mean([cumavg[1], cumavg[3], cumavg[5]])

    print("\n各时刻瞬时 L2（米）:")
    for k, v in enumerate(per_t):
        print(f"  t={0.5*(k+1):.1f}s  {v:.4f}")

    print("\n表格显示的（累积平均）:")
    for k, v in enumerate(cumavg):
        print(f"  {0.5*(k+1):.1f}s  {v:.4f}")

    print(f"\navg（论文定义）= {avg:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
