"""model.py —— 任务一：从 token 序列构建 N-gram 统计模型

【你要完成】build_ngram_model

模型就是一个嵌套字典：

    model[context][token] = 出现次数

其中 context 是长度为 n-1 的元组（n=1 时是空元组 ()），
内层是一个 collections.Counter。

例：tokens = ["我", "爱", "北京", "爱", "你"], n = 2
补全后序列：<START> 我 爱 北京 爱 你 <END>
得到的模型包含：
    ("<START>",) → {"我": 1}
    ("我",)      → {"爱": 1}
    ("爱",)      → {"北京": 1, "你": 1}     ← 同一个上下文可以有多个后继
    ("你",)      → {"<END>": 1}

先对照 docs/implement_guide.md 第 3 节理解数据结构，再动手。
"""
from collections import Counter, defaultdict

from tokenizer import START, END


def build_ngram_model(tokens, n):
    """统计 tokens 中所有 N-gram，返回模型字典。

    参数：
        tokens: token 列表（不含特殊符号，补全由你来做）
        n:      N-gram 的 N（n >= 1）

    返回：
        dict，键为 context（n-1 元组），值为 Counter（token -> 次数）

    提示（建议按顺序思考）：
        1. 先在 tokens 开头补 n-1 个 START、结尾补 1 个 END，
           得到补全后的序列 padded。
           想一想：为什么补 n-1 个而不是 1 个？（验收时会问）
        2. 用滑动窗口遍历 padded：对每个位置 i，
           context = tuple(padded[i : i + n - 1])，target = padded[i + n - 1]。
        3. 把 (context, target) 记入模型。
           defaultdict(Counter) 可以让"第一次见到某 context"时不用特判。
        4. 不要给 n == 1 写特判——正确的实现会天然涵盖它。
           写完检查一下：n=1 时 context 都是 ()，模型就是词频表。

    参考规模：约 10 行。
    """
    padded = ["<START>"] * (n - 1) + tokens + ["<END>"]
    model = defaultdict(Counter)
    for i in range(0,len(padded) - n + 1, 1):
        context = tuple(padded[i: i + n - 1])
        target = padded[i + n - 1] 
        model[context][target] += 1
    return model

def describe_model(model, n, top=5):
    """打印模型的整体统计信息（完整提供，帮助你看清"模型长什么样"）。"""
    total_contexts = len(model)
    total_entries = sum(len(counter) for counter in model.values())
    total_counts = sum(sum(counter.values()) for counter in model.values())

    print(f"N-gram 阶数 n = {n}")
    print(f"不同上下文数：{total_contexts}")
    print(f"(上下文, 后继) 条目总数：{total_entries}")
    print(f"统计的总 token 数：{total_counts}")
    if total_contexts:
        print(f"平均每个上下文有 {total_entries / total_contexts:.2f} 种不同后继")

    # 后继种类最多的上下文 = 模型最"犹豫"的地方
    most_uncertain = sorted(model.items(), key=lambda kv: len(kv[1]), reverse=True)[:top]
    print(f"\n后继最多样的 {top} 个上下文（模型最“犹豫”的地方）：")
    for context, counter in most_uncertain:
        preview = "、".join(f"{t}({c})" for t, c in counter.most_common(5))
        print(f"  {context} → {len(counter)} 种后继，最常见的：{preview}")
