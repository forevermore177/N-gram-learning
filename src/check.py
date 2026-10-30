"""check.py —— 自测脚本（完整提供，随时运行）

    python src/check.py

每完成一个任务就运行一次。所有测试用例都设计为可以手工复核
（见 docs/implement_guide.md 第 6 节），测试失败时请先自己手算一遍，
再对比你的程序输出。

注意：任务二、三的测试使用脚本内置的参考模型，不依赖你的任务一——
所以即使任务一还没完成，也可以单独验证任务二、三的正确性。
"""
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 让 Windows 老式终端也能正常显示
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tokenizer import START, END  # noqa: E402

results = []  # (名称, 状态, 备注)；状态 ∈ {"pass", "fail", "todo"}


def report(name, status, note=""):
    results.append((name, status, note))
    icon = {"pass": "✅ 通过", "fail": "❌ 失败", "todo": "⏳ 未完成"}[status]
    line = f"[{icon}] {name}"
    if note:
        line += f" —— {note}"
    print(line)


def reference_model(tokens, n):
    """脚本内置的参考模型，仅供测试任务二、三使用。"""
    model = defaultdict(Counter)
    padded = [START] * (n - 1) + list(tokens) + [END]
    for i in range(len(padded) - n + 1):
        model[tuple(padded[i:i + n - 1])][padded[i + n - 1]] += 1
    return dict(model)


# ---------------------------------------------------------------- 任务一
def check_build():
    name = "任务一：build_ngram_model 构建模型"
    try:
        from model import build_ngram_model
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return

    tokens = ["我", "爱", "北京", "爱", "你"]
    try:
        model = build_ngram_model(tokens, 2)
        assert model[(START,)] == Counter({"我": 1}), f'model[("<START>",)] 应为 {{"我": 1}}，实际 {model.get((START,))}'
        assert model[("爱",)] == Counter({"北京": 1, "你": 1}), f'model[("爱",)] 计数不对：{model.get(("爱",))}'
        assert model[("你",)] == Counter({END: 1}), f'model[("你",)] 应包含 <END>，实际 {model.get(("你",))}（忘记补 <END> 了？）'

        model1 = build_ngram_model(tokens, 1)
        assert model1[()]["爱"] == 2, f"n=1 时 model[()] 应为词频表，'爱' 出现 2 次，实际 {model1[()].get('爱')}"
        assert sum(model1[()].values()) == len(tokens) + 1, "n=1 时总计数应为 len(tokens)+1（含 <END>）"

        model3 = build_ngram_model(tokens, 3)
        assert model3[(START, START)] == Counter({"我": 1}), "n=3 时开头应补 2 个 <START>"
        assert model3[("爱", "北京")] == Counter({"爱": 1}), "n=3 滑动窗口计数不对"
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return
    except AssertionError as e:
        report(name, "fail", str(e))
        return
    report(name, "pass")


# ---------------------------------------------------------------- 任务二
def check_generate():
    name = "任务二：generate_text 生成文本"
    try:
        from generate import generate_text
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return

    try:
        # 确定性语料：每个上下文只有一个后继，生成结果必然恰好是 ["x", "y", "z"]
        model = reference_model(["x", "y", "z"], 2)
        out = generate_text(model, 2, length=10, seed=0)
        assert isinstance(out, list), "返回值应为 token 列表"
        assert START not in out and END not in out, "结果中不应包含 <START>/<END>"
        assert out == ["x", "y", "z"], f"确定性语料上应生成 ['x', 'y', 'z']，实际 {out}（采到 <END> 应停止）"

        # 长度上限与合法性
        model1 = reference_model(["a", "b", "a", "b"], 1)
        out1 = generate_text(model1, 1, length=7, seed=1)
        assert len(out1) <= 7, f"生成长度不应超过 length，实际 {len(out1)}"
        assert set(out1) <= {"a", "b"}, f"n=1 时只能生成词表中的 token，实际 {set(out1)}"

        # 可复现性：相同 seed 两次输出必须一致
        out_a = generate_text(model1, 1, length=20, seed=123)
        out_b = generate_text(model1, 1, length=20, seed=123)
        assert out_a == out_b, "相同 seed 的两次输出不一致（是不是用了全局 random 而不是 random.Random(seed)？）"
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return
    except AssertionError as e:
        report(name, "fail", str(e))
        return
    report(name, "pass")


# ---------------------------------------------------------------- 任务三（上）
def check_probability():
    name = "任务三(上)：token_probability 概率与平滑"
    try:
        from evaluate import token_probability
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return

    # 模型：("爱",) → {"北京": 1, "你": 1}；词表 {我, 爱, 北京, 你, <END>}，V = 5
    model = reference_model(["我", "爱", "北京", "爱", "你"], 2)
    V = 5
    try:
        p = token_probability(model, ("我",), "爱", V, smoothing=None)
        assert math.isclose(p, 1.0), f"P(爱|我) 无平滑应为 1.0，实际 {p}"
        p = token_probability(model, ("爱",), "北京", V, smoothing=None)
        assert math.isclose(p, 0.5), f"P(北京|爱) 无平滑应为 0.5，实际 {p}"
        p = token_probability(model, ("爱",), "北京", V, smoothing="laplace")
        assert math.isclose(p, 2 / 7), f"P(北京|爱) Laplace 应为 (1+1)/(2+5)=2/7，实际 {p}（分母忘加 |V| 了？）"
        p = token_probability(model, ("猫",), "爱", V, smoothing=None)
        assert p == 0, f"没见过的上下文，无平滑应返回 0，实际 {p}"
        p = token_probability(model, ("猫",), "爱", V, smoothing="laplace")
        assert math.isclose(p, 1 / V), f"没见过的上下文，Laplace 应退化为 1/V，实际 {p}"
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return
    except AssertionError as e:
        report(name, "fail", str(e))
        return
    report(name, "pass")


# ---------------------------------------------------------------- 任务三（下）
def check_perplexity():
    name = "任务三(下)：calculate_perplexity 困惑度"
    try:
        from evaluate import calculate_perplexity
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return

    model = reference_model(["x", "y", "z"], 2)
    V = 4  # {x, y, z, <END>}
    try:
        # 模型在“训练语料本身”上：每个后继概率都是 1，困惑度应恰好为 1
        ppl = calculate_perplexity(model, ["x", "y", "z"], 2, V, smoothing=None)
        assert math.isclose(ppl, 1.0, abs_tol=1e-9), \
            f"确定性模型在训练语料上的困惑度应为 1.0，实际 {ppl}（检查对数公式里的负号与 M）"

        # 测试集出现没见过的字：无平滑应为 inf，Laplace 应为有限值
        ppl_none = calculate_perplexity(model, ["x", "q"], 2, V, smoothing=None)
        assert ppl_none == float("inf"), f"遇到没见过的组合，无平滑困惑度应为 inf，实际 {ppl_none}"
        ppl_lap = calculate_perplexity(model, ["x", "q"], 2, V, smoothing="laplace")
        assert math.isfinite(ppl_lap) and ppl_lap > 1, f"Laplace 平滑后困惑度应为大于 1 的有限值，实际 {ppl_lap}"
    except NotImplementedError as e:
        report(name, "todo", str(e))
        return
    except AssertionError as e:
        report(name, "fail", str(e))
        return
    report(name, "pass")


def main():
    print("=" * 60)
    print("N-gram 项目自测")
    print("=" * 60)
    check_build()
    check_generate()
    check_probability()
    check_perplexity()

    print("-" * 60)
    n_pass = sum(1 for _, s, _ in results if s == "pass")
    if n_pass == len(results):
        print(f"全部 {n_pass}/{len(results)} 项通过！可以开始做对比实验了：")
        print("    python src/main.py compare")
        sys.exit(0)
    else:
        print(f"当前进度 {n_pass}/{len(results)}。失败/未完成的项请对照 docs/implement_guide.md 调试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
