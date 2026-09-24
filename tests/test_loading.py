"""逐管安全装载次序规划测试: python -m unittest discover -s tests

核心断言: 子集动态规划求得的全过程最小峰值 |U₁|、最优装载次序与达到最小
峰值的次序条数, 必须与全排列暴力枚举一致(同峰值按试管编号序列升序裁决);
逐步 |U₁| 复算正确、累计峰值单调且末步等于最终解; 超过 16 支明确拒绝;
规划过程不修改入参(只读)。
"""
import copy
import itertools
import math
import random
import time
import unittest

from app.loading import MAX_TUBES, plan_safe_loading


def _slots(n):
    return [
        {"index": i, "angle": 360.0 * i / n, "enabled": True}
        for i in range(n)
    ]


def _plan(rows, n_slots=12, radius=100.0):
    """rows: [(试管编号, 质量, 槽号)]。"""
    slots = _slots(n_slots)
    return {
        "id": 1,
        "name": "方案1",
        "created_at": "2026-09-24T00:00:00+00:00",
        "request": {"slot_count": n_slots, "slots": slots, "radius_mm": radius},
        "result": {
            "config": {"radius_mm": radius, "rpm": 4000.0},
            "placements": [
                {"tube_id": t, "slot": s, "mass": m, "angle": slots[s]["angle"]}
                for t, m, s in rows
            ],
        },
    }


def _brute(rows, n_slots=12, radius=100.0):
    """全排列参考: (最小峰值, 升序裁决后的编号序列, 最优次序条数)。"""
    info = {t: (m, s) for t, m, s in rows}
    best = None
    best_seq = None
    count = 0

    def _eq(a, b):
        return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))

    for perm in itertools.permutations([t for t, _, _ in rows]):
        ux = uy = 0.0
        peak = 0.0
        for t in perm:
            m, s = info[t]
            a = math.radians(360.0 * s / n_slots)
            ux += m * math.cos(a)
            uy += m * math.sin(a)
            peak = max(peak, radius * math.hypot(ux, uy))
        seq = tuple(perm)
        if best is None or peak < best - 1e-9 * max(1.0, abs(peak), abs(best)):
            best, best_seq, count = peak, seq, 1
        elif _eq(peak, best):
            count += 1
            best_seq = min(best_seq, seq)
    return best, list(best_seq), count


class TestSafeLoading(unittest.TestCase):
    def test_single_tube(self):
        r = plan_safe_loading(_plan([(7, 42.0, 3)]))
        self.assertEqual(r["order"], [7])
        self.assertEqual(r["optimal_order_count"], 1)
        self.assertAlmostEqual(r["peak_u1_mag_g_mm"], 4200.0, places=9)
        self.assertEqual(r["peak_step"], 1)
        self.assertEqual(len(r["steps"]), 1)
        st = r["steps"][0]
        self.assertEqual(st["step"], 1)
        self.assertEqual(st["tube_id"], 7)
        self.assertEqual(st["slot"], 3)
        self.assertAlmostEqual(st["angle_deg"], 90.0, places=9)
        self.assertAlmostEqual(st["u1_mag_g_mm"], 4200.0, places=9)
        self.assertAlmostEqual(st["cumulative_peak_g_mm"], 4200.0, places=9)
        self.assertIsNone(st["within_limit"])
        self.assertIsNone(st["excess_g_mm"])

    def test_opposite_pair_two_optimal_orders(self):
        # 等质量对径: 两支先后任意, 峰值都为 mR=5000; 终态完全平衡
        r = plan_safe_loading(_plan([(2, 50.0, 6), (1, 50.0, 0)]))
        self.assertEqual(r["order"], [1, 2])  # 编号序列升序
        self.assertEqual(r["optimal_order_count"], 2)
        self.assertTrue(r["tie_broken"])
        self.assertAlmostEqual(r["peak_u1_mag_g_mm"], 5000.0, places=9)
        self.assertLess(r["final_u1_mag_g_mm"], 1e-9)

    def test_three_tubes_120deg_lightest_first(self):
        # 50g@0°,40g@120°,30g@240°: 先轻后重, 峰值 3605.55 (优于先重 5000)
        plan = _plan([(1, 50.0, 0), (2, 40.0, 4), (3, 30.0, 8)])
        r = plan_safe_loading(plan)
        self.assertEqual(r["order"], [3, 2, 1])
        self.assertAlmostEqual(r["peak_u1_mag_g_mm"], 3605.551277, places=4)
        self.assertEqual(r["peak_step"], 2)
        # 逐步累计峰值单调不减, 末步 |U₁| = 最终布局 |U₁|
        peaks = [s["cumulative_peak_g_mm"] for s in r["steps"]]
        self.assertEqual(peaks, sorted(peaks))
        self.assertAlmostEqual(peaks[-1], r["peak_u1_mag_g_mm"], places=9)
        self.assertAlmostEqual(
            r["steps"][-1]["u1_mag_g_mm"],
            math.hypot(50.0 - 35.0, 0.0 + 40.0 * math.sin(math.radians(120.0))
                       + 30.0 * math.sin(math.radians(240.0))) * 100.0,
            places=6,
        )

    def test_random_against_brute_force(self):
        rng = random.Random(20260924)
        for _ in range(60):
            n_slots = rng.choice([3, 4, 5, 6, 8, 12])
            n_t = rng.randint(1, min(6, n_slots))
            slots = rng.sample(range(n_slots), n_t)
            tids = rng.sample(range(1, 60), n_t)
            rows = [
                (t, round(rng.uniform(10.0, 60.0), 3), s)
                for t, s in zip(tids, slots)
            ]
            r = plan_safe_loading(_plan(rows, n_slots=n_slots))
            best, seq, cnt = _brute(rows, n_slots=n_slots)
            self.assertAlmostEqual(
                r["peak_u1_mag_g_mm"], best, places=6,
                msg=f"peak mismatch rows={rows}",
            )
            self.assertEqual(r["order"], seq, f"order mismatch rows={rows}")
            self.assertEqual(
                r["optimal_order_count"], cnt, f"count mismatch rows={rows}")
            self.assertEqual(len(r["steps"]), n_t)
            cp = 0.0
            for st in r["steps"]:
                cp = max(cp, st["u1_mag_g_mm"])
                self.assertAlmostEqual(st["cumulative_peak_g_mm"], cp, places=9)

    def test_loading_limit_flags(self):
        plan = _plan([(1, 50.0, 0), (2, 40.0, 4), (3, 30.0, 8)])
        # 各步 |U₁|: 3000 -> 3605.55 -> 1732.05
        r = plan_safe_loading(plan, u1_limit_g_mm=3500.0)
        self.assertFalse(r["within_limit"])
        self.assertEqual(r["over_limit_steps"], [2])
        self.assertEqual(r["over_limit_count"], 1)
        self.assertTrue(r["steps"][0]["within_limit"])
        self.assertFalse(r["steps"][1]["within_limit"])
        self.assertTrue(r["steps"][2]["within_limit"])
        self.assertGreater(r["steps"][1]["excess_g_mm"], 0.0)
        self.assertEqual(r["steps"][0]["excess_g_mm"], 0.0)
        self.assertTrue(r["initial"]["within_limit"])
        # 限值不改变最优次序与峰值
        r2 = plan_safe_loading(plan, u1_limit_g_mm=100000.0)
        self.assertTrue(r2["within_limit"])
        self.assertEqual(r2["over_limit_steps"], [])
        self.assertEqual(r2["order"], r["order"])
        self.assertAlmostEqual(
            r2["peak_u1_mag_g_mm"], r["peak_u1_mag_g_mm"], places=9)
        # 边界: 限值恰等于峰值 -> 不超限(容差内)
        r3 = plan_safe_loading(plan, u1_limit_g_mm=r["peak_u1_mag_g_mm"])
        self.assertTrue(r3["within_limit"])

    def test_radius_changes_magnitude(self):
        plan = _plan([(1, 50.0, 0)], n_slots=12, radius=120.0)
        r = plan_safe_loading(plan)
        self.assertAlmostEqual(r["peak_u1_mag_g_mm"], 6000.0, places=9)
        self.assertEqual(r["radius_mm"], 120.0)

    def test_unplaced_tubes_excluded(self):
        plan = _plan([(1, 50.0, 0)])
        plan["result"]["placements"].append(
            {"tube_id": 9, "slot": None, "mass": 50.0, "angle": None})
        r = plan_safe_loading(plan)
        self.assertEqual(r["tube_count"], 1)
        self.assertEqual(r["unplaced_tube_ids"], [9])
        self.assertEqual(r["order"], [1])

    def test_max_16_accepted_and_timing(self):
        rows = [(i + 1, 50.0, i) for i in range(16)]
        t0 = time.perf_counter()
        r = plan_safe_loading(_plan(rows, n_slots=16))
        dt = time.perf_counter() - t0
        self.assertEqual(r["tube_count"], 16)
        self.assertEqual(len(r["steps"]), 16)
        self.assertLess(dt, 10.0)  # 2^16 子集 DP 应秒级完成
        self.assertLessEqual(r["optimal_order_count"], math.factorial(16))

    def test_read_only_inputs_unchanged(self):
        plan = _plan([(1, 50.0, 0), (2, 40.0, 4), (3, 30.0, 8)])
        snap = copy.deepcopy(plan)
        plan_safe_loading(plan, u1_limit_g_mm=3500.0)
        self.assertEqual(plan, snap)


class TestSafeLoadingValidation(unittest.TestCase):
    def test_more_than_16_tubes_rejected(self):
        rows = [(i + 1, 50.0, i % 18) for i in range(MAX_TUBES + 1)]
        with self.assertRaisesRegex(ValueError, str(MAX_TUBES)):
            plan_safe_loading(_plan(rows, n_slots=18))

    def test_empty_plan_rejected(self):
        with self.assertRaisesRegex(ValueError, "没有已放置"):
            plan_safe_loading(_plan([]))

    def test_duplicate_tube_rejected(self):
        plan = _plan([(1, 50.0, 0), (1, 40.0, 2)])
        with self.assertRaisesRegex(ValueError, "试管 1 重复"):
            plan_safe_loading(plan)

    def test_duplicate_slot_rejected(self):
        plan = _plan([(1, 50.0, 0), (2, 40.0, 0)])
        with self.assertRaisesRegex(ValueError, "槽位 0"):
            plan_safe_loading(plan)

    def test_bad_radius_rejected(self):
        plan = _plan([(1, 50.0, 0)])
        plan["result"]["config"]["radius_mm"] = 0.0
        plan["request"]["radius_mm"] = 0.0
        with self.assertRaisesRegex(ValueError, "半径"):
            plan_safe_loading(plan)

    def test_negative_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, "限值"):
            plan_safe_loading(_plan([(1, 50.0, 0)]), u1_limit_g_mm=-0.1)


if __name__ == "__main__":
    unittest.main()
