# OJ 题目材料：N-gram 计数与生成（Basic）

本目录是项目「24-NGram」Basic 部分（任务一~二）的 OnlineJudge 配套材料，供负责上传 OJ 的同事使用。

- `src/ngram.py`：**下发给学生**的代码框架（单文件，Python 3，含 `# TODO` 与英文引导注释），也可作为 OJ 附件发布；
- `testcases/`：全部评测数据，`N.in` / `N.out` 成对，共 17 个测试点，附 `config.json`（分组与时空限制）；
- 参考答案与数据生成器在 `ta/include/`（不下发）：`ngram_solution.py` 已对本目录全部测试点逐一验证通过（字节级一致），验证脚本为 `ta/include/verify_testcases.sh`。

## 题面（可直接粘贴到 OJ 题目描述）

**题目描述**

语言模型可以朴素到只用"数数"：统计"前 N-1 个 token 后面，每个 token 各出现过多少次"，就得到了一个 N-gram 模型。给定一段训练语料，请先建立这张计数表并回答若干计数查询（COUNT），再用它按计数加权随机采样，生成新的 token 序列（GEN）。

统计之前，语料序列需要先补齐：开头补 N-1 个特殊 token `<START>`，结尾补 1 个特殊 token `<END>`。例如语料为 `我 爱 北京 爱 你`、N=2 时，实际统计的序列是：

```text
<START> 我 爱 北京 爱 你 <END>
```

**输入格式**

```text
N Q
t1 t2 … tT
query_1
query_2
…
query_Q
```

- 第一行两个整数 `N Q`：N 为 N-gram 的阶（1 ≤ N ≤ 4），Q 为查询数（1 ≤ Q ≤ 100000）；
- 第二行为训练语料：T 个 token 以单个空格分隔（0 ≤ T ≤ 100000；token 是不含空白字符的 UTF-8 字符串，可能为中英文字、标点等，长度不超过 32 个字符）。T=0 时该行为空行；
- 随后 Q 行，每行一个查询，两种形式之一：
  - `COUNT c1 c2 … c_{N-1} w`：输出语料（补齐后）中上下文 `(c1,…,c_{N-1})` 后紧跟 `w` 的次数。**N=1 时格式为 `COUNT w`**（上下文为空）。上下文没出现过、或出现过但从没跟过 `w`，都输出 0。查询中可以出现 `<START>` / `<END>`（例如 N=2 时 `COUNT <START> 我` 是合法查询）；
  - `GEN L seed`：从上下文 `(<START>,)×(N-1)` 出发反复采样下一个 token，输出至多 L 个 token（1 ≤ L ≤ 1000），以单个空格分隔；0 ≤ seed < 2³¹。

**GEN 的采样规则（必须严格遵循，否则输出不可复现）**

1. 创建独立的随机数发生器：`rng = random.Random(seed)`（不要动全局 `random` 的状态）；
2. 每一步：取当前上下文的所有后继 token，按 **Python 默认字符串序**（即按 Unicode 码点逐字符比较，`sorted(...)` 的结果）排列为候选列表，各候选的权重为它的**整数计数**，然后恰好以
   `rng.choices(candidates, weights=counts, k=1)[0]`
   抽出下一个 token；
3. 若抽到 `<END>` 立即停止（`<END>` 不输出）；否则输出该 token，滑动窗口更新上下文（新上下文 = 旧上下文 + 新 token 的最后 N-1 个），继续；
4. 最多输出 L 个 token；若当前上下文不在模型中，也停止。

**输出格式**

每个查询输出一行：

- `COUNT`：一个非负整数；
- `GEN`：生成的 token 序列（至多 L 个），单个空格分隔；一个 token 都没有生成时输出空行。

**样例输入 1**

```text
2 6
我 爱 北京 爱 你
COUNT 爱 北京
COUNT 爱 你
COUNT <START> 我
COUNT 你 <END>
COUNT 我 你
COUNT 北京 爱
```

**样例输出 1**

```text
1
1
1
1
0
1
```

**样例输入 2**

```text
2 4
猫 追 狗 猫
COUNT 猫 追
COUNT 狗 猫
GEN 5 42
GEN 5 42
```

**样例输出 2**

```text
1
1
猫
猫
```

样例 2 说明：上下文 `（猫）` 的后继为 `{追: 1, <END>: 1}`，seed=42 时第一次就抽到 `<END>`，故只生成一个 token；同一 seed 的两次 GEN 输出完全相同。

**数据范围与提示**

- 1 ≤ N ≤ 4，1 ≤ Q ≤ 100000，0 ≤ T ≤ 100000，1 ≤ L ≤ 1000，0 ≤ seed < 2³¹；
- 语言：Python 3，仅标准库。建议实现复杂度：建模 O(T)，每次 COUNT O(1)；
- 评测为全文精确比较：注意每个查询恰好一行、行内 token 间只有一个空格、文件末尾有换行；
- GEN 的确定性依赖 CPython 的 `random` 模块实现。本题的 `.out` 文件由 **Python 3.10.12** 的参考答案生成；若评测机的 Python 版本与此不同导致 GEN 测试点不一致，请在评测机环境用 `ta/include/gen_testcases.py` 重新生成数据。

## 上传与打包说明

1. 在 OJ 后台新建题目（建议标题「N-gram 计数与生成」），语言限定 Python 3，粘贴上方题面；
2. 测试数据用「数据 (zip)」上传，**格式要求严格**（见 OJ 官方帮助页「如何创建数据包？」/ `/OnlineJudge/help/admin/data-format`）：
   - **压缩包文件名必须是 `题号.zip`**（例如本题为 `3262.zip`），题号是数据库主键 id，不是标题；
   - **包内顶层必须是以题号命名的文件夹**，所有文件放在其中：`题号.zip/题号/1.in`、`题号.zip/题号/1.out`、…、`题号.zip/题号/config.json`（结构与 A+B 题目的数据包 `1000.zip/1000/…` 一致）；
   - `config.json` 必须在该文件夹内，缺失或路径不对会报"需要 json 文件"之类的错误。
   
   在本目录执行（把 `<ID>` 换成实际题号，**两处都要换**）：

   ```bash
   cd testcases && mkdir -p <ID> && cp *.in *.out config.json <ID>/ && zip -r ../<ID>.zip <ID> && rm -rf <ID>
   ```

   若使用「数据（GUI）」上传，则逐点上传 `.in`/`.out` 并按 `config.json` 手工建组：simple（1–4）20 分、complex（5–10）30 分、large（11–14）40 分、robust（15–17）10 分；所有测试点时限 2000 ms，内存均 256 MiB，无需 SPJ（`SPJ: 0`）；
3. 如需重新生成或调整测试点，运行 `python3 ta/include/gen_testcases.py include/testcases`（生成器与参考答案相互独立，并用 `bash ta/include/verify_testcases.sh` 交叉验证：逐点字节级 diff 并计时）。

## 测试点一览

| 编号 | 分组 | 考查点 |
| --- | --- | --- |
| 1 | simple | README 小例子（`我 爱 北京 爱 你`，N=2），6 个 COUNT 全部可手算，含 `<START>`/`<END>` 与未出现组合 |
| 2 | simple | N=1 退化为词频统计（含 `<END>`=1、`<START>`=0），附一次小 GEN |
| 3 | simple | 确定性语料（每上下文唯一后继）：GEN 精确复现原文 + 长度上限截断 |
| 4 | simple | 单 token 语料 N=1：GEN 首采即 `<END>` 输出空行 / 采一次后停止 |
| 5 | complex | N=3：双 `<START>` 上下文、重复上下文计数、`<END>` 查询、小 GEN |
| 6 | complex | N=4：三 `<START>` 前缀、同一上下文两个后继（其一为 `<END>`） |
| 7 | complex | 中英文与标点混合 token；同 seed 两次 GEN 结果必须一致 |
| 8 | complex | GEN 撞长度上限 vs 提前遇 `<END>` 停止（同一模型两个 seed） |
| 9 | complex | N=3 可复现性：同 seed 两行完全相同，不同 seed 输出不同 |
| 10 | complex | COUNT 边界：见过的上下文+没见过的词、没见过的上下文均返回 0 |
| 11 | large | 《红楼梦》字符语料×15（84450 token），N=2，62 个 COUNT + 2 个 GEN |
| 12 | large | 同上语料，N=3，42 个 COUNT + GEN 50 |
| 13 | large | 鲁迅语料×20（46320 token），N=4，42 个 COUNT + GEN 40 |
| 14 | large | 《红楼梦》语料 N=2 长生成：两次 GEN 500 + COUNT |
| 15 | robust | 语料长度（2）小于 N（4）：全靠 `<START>` 补齐；GEN 确定性输出 |
| 16 | robust | N=1 单一 token：GEN 连采 20 次撞上限 / 提前 `<END>`；`COUNT <START>`=0 |
| 17 | robust | 特殊符号当普通 token 查询：`COUNT <START> <START>`、`<END>` 不作上下文；GEN L=1 |

参考答案本机实测：全部 17 点通过，最慢测试点（12 号，N=3 大语料）约 120 ms，远低于 2000 ms 时限。
