"""已保存配平方案的逐管安全装载次序规划(只读, 不改变求解与保存格式)。

问题模型
--------
选定一条已保存方案, 以**空转子**为起点, 每一步只装入一支试管, 全部装完
后转子状态与该方案**完全一致**(各管质量、槽位角度与半径均只读取自方案,
最终槽位不变)。第 k 步装入后, 已装入集合 S_k 的一阶不平衡量:

    U₁(S_k) = R · Σ_{i ∈ S_k} m_i · e^{iθ_i}

装载过程关心的不是终态(终态即原方案), 而是全过程出现过的最大一阶不平衡量:

    峰值 |U₁| = max_{k=1..n} |U₁(S_k)|

(空转子 |U₁| = 0。) 要求精确找到使该峰值最小的装管次序。

子集动态规划(精确求序, n ≤ 16)
------------------------------
集合 S 的瞬时不平衡量只取决于 S 本身, 与装入先后无关; 而 S 作为某条装载
路径的前缀出现时, 其"此前峰值"可以优化。定义:

    peak(∅) = 0
    peak(S)  = max( |U₁(S)|, min_{i ∈ S} peak(S \\ {i}) )

即最后装入的试管 i 取为"使前缀峰值最小"的那一支。该递推枚举 n 支试管的
全部 2ⁿ 个子集, 每个子集至多尝试 n 支末管, O(n·2ⁿ)(n = 16 时约 105 万次
比较), 精确求出全过程最小峰值; 不使用贪心或近似。

同成本裁决
----------
达到最小峰值 P* 的次序可能不止一条。最优次序等价于在"安全集合格"
(所有满足 |U₁(S)| ≤ P* 的子集 S)上从 ∅ 到全集的一条逐加一管的路径:

- 前向路径计数 ways[S] = Σ_{i∈S, S 安全} ways[S \ {i}](ways[∅]=1),
  ways[全集] 即达到最小峰值的次序条数;
- 后向可达 ok[S]: 从 S 只经过安全集合能否到达全集。

字典序最小的次序用贪心恢复: 每一步在剩余试管中选编号最小、且加入后集合仍后向
可达(存在补全为最优次序的可能)的试管。

装载阶段限值
------------
调用方可给出一阶不平衡量限值 L(g·mm); 限值逐**中间状态**判定: 每个装入
步骤后只要 |U₁(S_k)| > L 即记为超限步骤(终态 |U₁| 同样参与判定)。限值
不影响最优次序的求解, 只用于标注超限步骤与总体合格判定。
"""
from __future__ import annotations

import math
from typing import Any, Optional

# 与 app/batch.py / app/weighing.py 一致的数值比较风格
_NUM_TOL = 1e-9
MAX_TUBES = 16


def plan_safe_loading(plan: dict[str, Any],
                      u1_limit_g_mm: Optional[float] = None) -> dict[str, Any]:
    """对一条已保存方案精确规划逐管安全装载次序(只读)。

    参数
    ----
    plan: GET /api/plan/{id} 返回结构(含 id/name/request/result),
        落槽、角度与半径全部只读取自该方案。
    u1_limit_g_mm: 装载阶段一阶不平衡量限值(g·mm), 可空; 非空时必须为
        有限非负数。逐中间状态判定, 不影响最优次序。

    返回
    ----
    含最小峰值、峰值所在步骤、最优次序、达到最小峰值的次序条数、逐步
    装入清单(当步 |U₁|、累计峰值、超限标注)的字典; 不修改方案数据。
    """
    result = plan.get("result") or {}
    placements = result.get("placements") or []
    config = result.get("config") or {}
    radius_mm = float(config.get("radius_mm")
                      or (plan.get("request") or {}).get("radius_mm", 100.0))
    if not (radius_mm > 0.0) or not math.isfinite(radius_mm):
        raise ValueError("转子半径必须为正有限数。")

    limit: Optional[float] = None
    if u1_limit_g_mm is not None:
        limit = float(u1_limit_g_mm)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError("装载阶段一阶不平衡量限值必须为有限非负数。")

    unplaced_ids: list[int] = []
    placed: list[dict[str, Any]] = []
    seen_tubes: set[int] = set()
    seen_slots: set[int] = set()
    for p in placements:
        tid = int(p["tube_id"])
        if tid in seen_tubes:
            raise ValueError(f"摆放明细中试管 {tid} 重复。")
        seen_tubes.add(tid)
        if p.get("slot") is None:
            unplaced_ids.append(tid)
            continue
        slot = int(p["slot"])
        if slot in seen_slots:
            raise ValueError(f"摆放明细中槽位 {slot} 被重复占用。")
        seen_slots.add(slot)
        if p.get("angle") is None:
            raise ValueError(f"试管 {tid} 的摆放结果缺少槽位角度。")
        mass = float(p["mass"])
        if not (mass > 0.0) or not math.isfinite(mass):
            raise ValueError(f"试管 {tid} 的质量必须为正有限数。")
        placed.append({
            "tube_id": tid,
            "mass": mass,
            "slot": slot,
            "angle_deg": float(p["angle"]),
        })

    n = len(placed)
    if n == 0:
        raise ValueError("该方案没有已放置的试管, 无法规划逐管装载次序。")
    if n > MAX_TUBES:
        raise ValueError(
            f"逐管安全装载最多支持 {MAX_TUBES} 支已放置试管"
            f"(当前 {n} 支); 子集动态规划状态数为 2^n, "
            f"超过 {MAX_TUBES} 支明确拒绝, 不做近似计算。"
        )

    # 每支试管的一阶质径积向量(g·mm); 顺序即 DP 位序
    placed.sort(key=lambda t: t["tube_id"])
    vx = [0.0] * n
    vy = [0.0] * n
    for i, t in enumerate(placed):
        theta = math.radians(t["angle_deg"])
        vx[i] = radius_mm * t["mass"] * math.cos(theta)
        vy[i] = radius_mm * t["mass"] * math.sin(theta)

    size = 1 << n
    full = size - 1

    # 各子集的单位质量复数和(直接累加), 供 |U₁(S)| 与逐步复算使用
    sx = [0.0] * size
    sy = [0.0] * size
    mag = [0.0] * size
    for mask in range(1, size):
        lb = mask & -mask
        i = lb.bit_length() - 1
        prev = mask ^ lb
        sx[mask] = sx[prev] + vx[i]
        sy[mask] = sy[prev] + vy[i]
        mag[mask] = math.hypot(sx[mask], sy[mask])

    # ---- 1) 值 DP: peak[mask] = 以 mask 为已装入集合时可达到的最小峰值 ----
    tol = _NUM_TOL
    peak = [0.0] * size
    for msk in range(1, size):
        best_prev: Optional[float] = None
        bits = msk
        m_here = mag[msk]
        while bits:
            lb = bits & -bits
            pv = peak[msk ^ lb]
            if best_prev is None or pv < best_prev - tol * max(
                    1.0, abs(pv), abs(best_prev)):
                best_prev = pv
            bits ^= lb
        assert best_prev is not None
        peak[msk] = m_here if m_here > best_prev + tol * max(
            1.0, m_here, abs(best_prev)) else best_prev

    optimal_peak = peak[full]

    # 安全集合: |U₁(mask)| ≤ P* 的集合(空集与全集必然安全)
    safe = bytearray(size)
    for msk in range(size):
        mg = mag[msk]
        if mg <= optimal_peak + tol * max(1.0, mg, abs(optimal_peak)):
            safe[msk] = 1

    # ---- 2) 路径计数: 从 ∅ 沿安全集合逐加一管到达 mask 的路径数 ----
    nways = [0] * size
    nways[0] = 1
    for msk in range(1, size):
        if not safe[msk]:
            continue
        total = 0
        bits = msk
        while bits:
            lb = bits & -bits
            total += nways[msk ^ lb]
            bits ^= lb
        nways[msk] = total
    optimal_count = nways[full]

    # ---- 3) 后向可达: mask 只经过安全集合能否到达全集 ----
    bwd = bytearray(size)
    bwd[full] = safe[full]
    for msk in range(full - 1, -1, -1):
        if not safe[msk]:
            continue
        rem = full ^ msk
        bits = rem
        while bits:
            lb = bits & -bits
            if bwd[msk | lb]:
                bwd[msk] = 1
                break
            bits ^= lb

    # ---- 4) 字典序最小次序: 每步取编号最小、加入后仍后向可达的试管 ----
    order_idx: list[int] = []
    msk = 0
    while msk != full:
        rem = full ^ msk
        chosen: Optional[int] = None
        # placed 已按试管编号升序, 位序即编号升序, 取首个可行位即可
        bits = rem
        while bits:
            lb = bits & -bits
            if bwd[msk | lb]:
                chosen = lb.bit_length() - 1
                break
            bits ^= lb
        assert chosen is not None
        order_idx.append(chosen)
        msk |= 1 << chosen
    order_ids = [placed[i]["tube_id"] for i in order_idx]

    # 逐步装载清单: 空转子起步, 每步一支, 累计峰值 = 截至当步的最大 |U₁|
    steps: list[dict[str, Any]] = []
    over_steps: list[int] = []
    msk = 0
    cumulative_peak = 0.0
    peak_step = 0
    peak_xy = (0.0, 0.0)
    for k, i in enumerate(order_idx, start=1):
        msk |= 1 << i
        ux, uy = sx[msk], sy[msk]
        u_mag = mag[msk]
        if u_mag > cumulative_peak + tol * max(1.0, u_mag, cumulative_peak):
            cumulative_peak = u_mag
            peak_step = k
            peak_xy = (ux, uy)
        within = None if limit is None else u_mag <= limit + tol * max(
            1.0, u_mag, limit)
        if within is False:
            over_steps.append(k)
        t = placed[i]
        steps.append({
            "step": k,
            "tube_id": t["tube_id"],
            "mass_g": t["mass"],
            "slot": t["slot"],
            "angle_deg": t["angle_deg"],
            "u1_mag_g_mm": u_mag,
            "u1_angle_deg": math.degrees(math.atan2(uy, ux)) % 360.0,
            "u1_x_g_mm": ux,
            "u1_y_g_mm": uy,
            "cumulative_peak_g_mm": cumulative_peak,
            "within_limit": within,
            "excess_g_mm": (None if limit is None
                            else max(0.0, u_mag - limit)),
        })

    peak_full = optimal_peak
    final_u1 = mag[full]
    peak_x, peak_y = peak_xy
    within_all = None if limit is None else len(over_steps) == 0
    return {
        "method": (
            "子集动态规划精确求全过程峰值 |U₁| 最小的逐管装载次序"
            "(空转子起步, 每步一支, 最终槽位不变); "
            "同峰值按试管编号序列升序裁决"
        ),
        "radius_mm": radius_mm,
        "tube_count": n,
        "peak_u1_mag_g_mm": peak_full,
        "peak_u1_angle_deg": math.degrees(
            math.atan2(peak_y, peak_x)) % 360.0 if peak_full > 0.0 else 0.0,
        "peak_step": peak_step,
        "optimal_order_count": optimal_count,
        "tie_broken": optimal_count > 1,
        "order": order_ids,
        "steps": steps,
        "initial": {
            "step": 0,
            "tube_id": None,
            "u1_mag_g_mm": 0.0,
            "u1_angle_deg": 0.0,
            "u1_x_g_mm": 0.0,
            "u1_y_g_mm": 0.0,
            "cumulative_peak_g_mm": 0.0,
            "within_limit": (None if limit is None else True),
        },
        "final_u1_mag_g_mm": final_u1,
        "u1_limit_g_mm": limit,
        "within_limit": within_all,
        "over_limit_steps": over_steps,
        "over_limit_count": len(over_steps),
        "unplaced_tube_ids": sorted(unplaced_ids),
        "note": "逐管安全装载为只读规划结果, 未修改任何已保存方案。",
    }
