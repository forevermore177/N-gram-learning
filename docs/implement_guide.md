# N-gram 项目代码编写指南

本文档是 `README.md` 的配套实现指南：先读 README 了解"做什么"，再读本文档了解"怎么写"。所有代码只需 Python 标准库。

## 1. 项目目标与原则

完成三个相互衔接的功能：

1. 从文本统计出 N-gram 模型（一个"上下文 → 后继计数"的字典）；
2. 从模型按概率采样生成新文本；
3. 用困惑度定量评估模型，并用平滑处理"没见过"的情况。

实现时以简单、清楚、可调试为先：

- 每个函数只完成一个明确任务；
- 优先用 `collections.Counter` / `dict` 等内置结构，不自己造数据结构；
- 随机性只通过 `seed` 参数控制，保证实验可复现；
- 暂不考虑大规模性能优化、多进程、图形界面——示例语料规模下朴素实现足够快。

## 2. 推荐架构

```text
src/
├── check.py       （完整提供）自测：用人工算过的小例子逐项验证
├── main.py        （完整提供）命令行入口，只管参数解析与输入输出
├── tokenizer.py   （完整提供）文本 → token 序列；语料文件加载
├── model.py       （任务一）tokens → N-gram 计数模型
├── generate.py    （任务二）model → 新 token 序列
└── evaluate.py    （任务三）model + 测试集 → 困惑度
```

依赖关系（箭头 = import）：

```text
check.py ──→ model.py / generate.py / evaluate.py
main.py  ──→ tokenizer.py / model.py / generate.py / evaluate.py
```

`tokenizer.py` 不依赖任何其他文件；三个任务文件互不依赖。你可以按任意顺序完成三个任务，但建议按 一 → 二 → 三 的顺序，因为后一个任务的"玩起来"需要前一个。

## 3. 数据表示

### 3.1 token

一个 token 就是一个字符串。中文按字切分时，`"星"`、`"，` 都是 token；英文按词切分时，`"the"` 是 token。另有两个永不出现在原文中的特殊 token：

```python
START = "<START>"   # 句首填充
END   = "<END>"     # 文本结束
```

### 3.2 模型

模型就是一个嵌套字典：

```python
model[context][token] = count
```

- `context`：长度为 `n-1` 的元组，如 `("仰望",)` 是 bigram 的上下文（n=2）；`()` 是 unigram 的上下文（n=1）。
- 内层是一个 `Counter`：记录该上下文后每个 token 出现的次数。

例如 n=2 时 `model[("爱",)] == Counter({"北京": 1, "你": 1})` 表示"爱"字后面见过一次"北京"、一次"你"。

**概率从计数来**：`P(token | context) = model[context][token] / sum(model[context].values())`（无平滑时）。

### 3.3 补全（padding）

统计和评估时，token 序列都要先补全：

```text
原始:    我 爱 北 京
补全:    <START> ... <START>  我 爱 北 京  <END>
          └── n-1 个 ──┘
```

为什么补 n-1 个 `<START>`？因为这样第一个真实 token 也拥有长度恰为 n-1 的上下文，序列开头和中间被一视同仁。为什么要有 `<END>`？因为模型需要学会"什么时候停下来"，否则生成永远不会自然结束。

## 4. 函数契约

四个 TODO 函数的输入输出约定如下（签名已写好，**不要改签名**，`check.py` 和 `main.py` 依赖它们）：

### `build_ngram_model(tokens: list[str], n: int) -> dict`（model.py）

- 返回上述嵌套字典（用 `defaultdict(Counter)` 最方便，返回普通 dict 亦可）。
- 必须处理 `n == 1`：此时每个上下文都是空元组 `()`，模型退化为词频统计——**不需要为它特判**，正确的滑动窗口实现天然涵盖这种情况。写完想想为什么。
- 参考规模：约 10 行。

### `generate_text(model, n, length=100, seed=None, top_k=None, temperature=1.0) -> list[str]`（generate.py）

- 从 `context = ("<START>",) * (n-1)` 出发（n=1 时是 `()`）；
- 每步从 `model[context]` 中**按计数加权**随机采样下一个 token（`random.choices(population, weights=..., k=1)`）；
- 采到 `<END>` 或已生成 `length` 个 token 时停止；
- **返回生成的 token 列表，不含 `<START>` 与 `<END>`**；
- `seed` 用于构造独立的随机数发生器（`rng = random.Random(seed)`），同样的 seed 必须产生同样的输出——不要在函数里碰全局随机种子；
- 若当前 context 不在模型中（理论上罕见），直接结束生成即可；
- `top_k` / `temperature` 是给进阶部分扩展预留的参数，基础版直接忽略。
- 参考规模：约 15 行。

### `token_probability(model, context, token, vocab_size, smoothing=None) -> float`（evaluate.py）

返回 `P(token | context)`：

| 情况 | `smoothing=None`（最大似然） | `smoothing="laplace"` |
| --- | --- | --- |
| context 见过，token 见过 | `count / total` | `(count + 1) / (total + V)` |
| context 见过，token 没见过 | `0` | `1 / (total + V)` |
| context 没见过 | `0` | `1 / V`（退化为均匀分布） |

注意第三行：上下文整个没见过时，Laplace 平滑下分母也没有意义了，约定退回均匀分布。

### `calculate_perplexity(model, test_tokens, n, vocab_size, smoothing=None) -> float`（evaluate.py）

- 先对 `test_tokens` 做与训练时一致的补全，再对每个位置调用 `token_probability`；
- 若某位置概率为 0：**直接返回 `float("inf")`**。不用觉得可惜——"无平滑的模型在测试集上原地爆炸"本身就是最重要的实验现象，让它发生；
- 否则累加对数概率，返回 `exp(-sum_log_prob / M)`，M 为参与求和的 token 数（含 `<END>`，不含 `<START>`）；
- 建议顺便统计零概率出现的次数，用 `print` 告知用户（这对实验分析和课堂展示有用）。

## 5. 推荐实现顺序

1. 通读 `tokenizer.py`，在 Python 里手动跑一遍：`tokenize("你好，世界", "char")` 看看返回什么；
2. 完成任务一，**先对着 README 里"我 爱 北京 爱 你"的小例子手算**，再跑 `check.py` 验证；
3. 完成任务二，用同一个小例子手推生成过程（每一步的候选和权重各是什么）；
4. 完成任务三，先写 `token_probability` 再写 `calculate_perplexity`；
5. 每一步都用小例子验证，**不要等全部写完才运行**——`check.py` 就是为这个准备的。

## 6. 测试建议

`check.py` 的全部用例都可以手算复核，调试时建议自己也这样做：

- **任务一**：`["我","爱","北京","爱","你"]`、n=2 的模型，`model[("爱",)]` 应为 `{"北京":1,"你":1}`；n=1 时 `model[()]["爱"]` 应为 2。
- **任务二**：用 `["x","y","z"]` 建 bigram 模型后，每个上下文都只有一个后继，生成结果必然恰好是 `["x","y","z"]`（采到 `<END>` 停止）。这种"确定性语料"是调试采样逻辑的利器。
- **任务三**：模型对训练语料本身的困惑度（无平滑）应当恰好是 1.0（每个后继概率都是 1）；把测试集换成包含没见过的字，无平滑应为 `inf`，Laplace 应有限。

调试时可观察的不变量：

- 所有内层 Counter 的计数都是正整数；
- 无平滑时，同一 context 下所有 token 的概率之和为 1；
- 同样的 seed 调用两次 `generate_text`，输出完全相同；
- `len(生成结果) <= length`，且结果中不含 `<START>` / `<END>`。

## 7. 常见错误

1. **给 n=1 特判并把模型建错**。n=1 的上下文就是 `()`，滑动窗口统一处理即可。
2. **补全只补一个 `<START>`**。要补 n-1 个，否则序列开头的 token 没有完整上下文。
3. **忘记补 `<END>`**。模型永远学不会结束，生成只能撞长度上限。
4. **评估时打乱语料再切分**。文本是有序数据，打乱会彻底破坏"用前文预测后文"的意义；直接按 8:2 顺序切。
5. **对数概率用错公式**。PPL 定义里是 `-1/M × Σ ln P`，漏掉负号会得到小于 1 的"困惑度"。
6. **Laplace 平滑时分母忘加 `|V|`**。加一平滑的本质是把一部分概率质量均分给全部词表。
7. **在 `generate_text` 里调 `random.seed(seed)`**。这会污染全局随机状态；应使用 `random.Random(seed)` 创建独立发生器。
8. **把 `<START>` 当成可生成的 token**。`<START>` 只能出现在上下文里，绝不能作为采样结果（想想为什么它不会出现在任何 Counter 里——除非你补全写错了）。
9. **词表忘了包含 `<END>`**。`<END>` 是合法的采样目标，词表大小 V 应包含它。

## 8. 完成标准

满足以下条件即可认为基础版本完成：

- `check.py` 四项全部通过；
- 对任意 UTF-8 语料、任意 1 ≤ n ≤ 4，三个任务函数都能正确工作；
- `main.py compare` 能输出完整的对比表（生成样例 + 两组困惑度）并存盘；
- 所有实验可用固定 seed 复现；
- 能逐行向助教解释自己写的每一行代码。

在这些功能稳定前，不建议加图形界面、中文分词库或性能优化。对入门项目而言，容易阅读、能够解释、结果正确，比功能数量更重要。
