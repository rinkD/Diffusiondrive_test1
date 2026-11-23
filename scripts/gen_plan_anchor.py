# -*- coding: utf-8 -*-
"""
生成 plan anchor（规划锚点），形状 (3, K, 6, 2)。

官方 tools/kmeans/kmeans_plan.py 只做全局聚类，生成 (K, 6, 2) 并存成
kmeans_plan_vocab_{K}.npy。但配置里 plan_anchor 要的是 (3, K, 6, 2)：
3 条导航指令 × K 个模态 × 6 个未来帧 × 2 个坐标。

所以这里按 gt_ego_fut_cmd（导航指令）分成三组，每组单独聚类再堆起来。

用法：
    cd <DiffusionDrive 根目录>
    python /path/to/gen_plan_anchor.py

前提：data/infos/nuscenes_infos_train.pkl 已生成。
"""
import numpy as np
import mmcv
from sklearn.cluster import KMeans

# 每条导航指令聚多少个模态。config 里 ego_fut_mode=6，所以要 6
K = 6

# 索引文件路径。mini 版本的话文件在 data/infos/mini/ 下，
# 或者事先建好 data/infos/nuscenes_infos_train.pkl 的软链接
INFO_PATH = "data/infos/nuscenes_infos_train.pkl"

# 未来帧数，和 config 里的 ego_fut_ts 一致
EGO_FUT_TS = 6


def main():
    data = mmcv.load(INFO_PATH)
    infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
    print(f"索引里共 {len(infos)} 个样本")

    # 按导航指令分三桶。gt_ego_fut_cmd 是 one-hot，
    # argmax 出来 0=右转、1=左转、2=直行（converter 里的约定）
    buckets = [[], [], []]
    skipped = 0
    for info in infos:
        mask = info["gt_ego_fut_masks"]
        # 轨迹不完整的样本不要，否则会把缺口聚进锚点里
        if not mask.sum() == EGO_FUT_TS:
            skipped += 1
            continue
        # 存的是增量，要先累加成绝对位置再聚类
        traj = info["gt_ego_fut_trajs"].cumsum(axis=-2)
        cmd = int(info["gt_ego_fut_cmd"].astype(np.int32).argmax(axis=-1))
        buckets[cmd].append(traj)

    print(f"跳过 {skipped} 个掩码不全的样本")
    print(f"每组的样本数: {[len(b) for b in buckets]}")

    # 兜底用的全局聚类结果。mini 数据集里可能某一组一个样本都没有
    # （实测 mini 没有左转样本，命令 0 是空的），那种情况只能拿全局的顶上
    all_traj = np.concatenate([t for b in buckets for t in b], axis=0)
    fallback = None

    clusters = []
    for i in range(3):
        group = buckets[i]
        if len(group) >= K:
            flat = np.concatenate(group, axis=0).reshape(-1, EGO_FUT_TS * 2)
            centers = KMeans(n_clusters=K, random_state=0, n_init=10).fit(flat).cluster_centers_
        else:
            print(f"  命令 {i} 只有 {len(group)} 个样本，不够聚 {K} 类，改用全局聚类")
            if fallback is None:
                flat_all = all_traj.reshape(-1, EGO_FUT_TS * 2)
                fallback = KMeans(n_clusters=K, random_state=0, n_init=10).fit(flat_all).cluster_centers_
            centers = fallback
        clusters.append(centers.reshape(K, EGO_FUT_TS, 2))

    # 堆成 (3, K, 6, 2)
    out = np.stack(clusters, axis=0).astype(np.float32)
    print(f"plan_anchor 形状: {out.shape}  {out.dtype}")

    np.save("data/kmeans/kmeans_plan_6.npy", out)
    print("已保存 data/kmeans/kmeans_plan_6.npy")


if __name__ == "__main__":
    main()
