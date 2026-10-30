"""generate.py —— 任务二：从 N-gram 模型生成文本

【你要完成】generate_text

生成就是反复"按概率掷骰子"：

    当前上下文 context  ──查模型──▶  若干候选 token 及其次数
                                     │  按次数作为权重随机采样
                                     ▼
                              下一个 token，加入序列
                                     │  滑动窗口更新 context
                                     └── 重复，直到采到 <END> 或达到 length

建议先用确定性语料调试：tokens = ["x", "y", "z"] 建立的 bigram 模型中
每个上下文只有一个后继，生成结果必然恰好是 ["x", "y", "z"]。
"""
import random

from tokenizer import START, END


def generate_text(model, n, length=100, seed=None, top_k=None, temperature=1.0):
    """用模型生成一段新文本。

    参数：
        model:       build_ngram_model 的返回值
        n:           建模时用的 N（必须与模型一致）
        length:      最多生成多少个 token
        seed:        随机种子；相同的 seed 必须产生相同的输出（实验要可复现！）
        top_k:       【进阶部分扩展预留】只在概率最高的 k 个候选中采样；None 表示不限制
        temperature: 【进阶部分扩展预留】温度采样参数；基础版忽略即可

    返回：
        生成的 token 列表，不含 <START> 和 <END>。

    提示（建议按顺序思考）：
        1. 用 rng = random.Random(seed) 创建独立的随机数发生器。
           不要调用全局的 random.seed()——那会污染别人的随机状态。
        2. 初始上下文：context = (START,) * (n - 1)。n=1 时它是 ()，不用特判。
        3. 每一步：查 model[context] 得到 Counter，
           用 rng.choices(population, weights=..., k=1) 按次数加权采样一个。
           （注意 choices 返回的是列表，记得取出里面的元素。）
        4. 采到 END 就停止；否则把 token 加入结果，并滑动窗口更新 context：
           context = (旧的 context + 新 token) 的最后 n-1 个。
        5. 如果 context 不在模型里（理论上很少见），直接 break。
        6. 循环最多执行 length 次。

    参考规模：约 15 行。
    """
    rng = random.Random(seed)
    context = (START,) * (n - 1)
    text = []
    for i in range(length):
        if context not in model:
            break
        Counter = model[context]
        next_token = rng.choices(list(Counter.keys()), weights = list(Counter.values()),k = 1)[0]
        if next_token == END :
            break
        if n == 1:
            context = ()
        else:
            context = (context + (next_token,))[-(n - 1):]
        text.append(next_token)
    return text
