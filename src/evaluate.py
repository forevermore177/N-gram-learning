"""evaluate.py —— 任务三：用困惑度（Perplexity）评估模型，用平滑处理"没见过"

【你要完成】token_probability、calculate_perplexity

困惑度的直觉：模型预测下一个 token 时，平均相当于在多少个
等可能候选中"犹豫"。越低说明模型越"懂"这份文本。

    PPL = exp( -1/M * Σ ln P(w_i | context_i) )

文本是有序数据，切分训练/测试集时【不能打乱】——直接用前 80% 训练、
后 20% 测试（split_train_test 已经替你做好）。
"""
import math

from tokenizer import START, END


def split_train_test(tokens, train_ratio=0.8):
    """按顺序把 token 序列切成训练集和测试集（完整提供）。"""
    cut = int(len(tokens) * train_ratio)
    return tokens[:cut], tokens[cut:]


def build_vocab(tokens):
    """根据训练集构建词表（完整提供）。注意 END 是合法的生成目标，必须在词表中。"""
    return set(tokens) | {END}


def token_probability(model, context, token, vocab_size, smoothing=None):
    """返回 P(token | context)。

    参数：
        model:      build_ngram_model 的返回值
        context:    上下文元组（n=1 时是 ()）
        token:      要评估概率的 token
        vocab_size: 词表大小 |V|（见 build_vocab）
        smoothing:  None 表示无平滑（最大似然估计）；"laplace" 表示加一平滑

    三种情况（对照 docs/implement_guide.md 第 4 节的表格）：
        1. context 见过、token 也见过：
           无平滑  count / total
           laplace (count + 1) / (total + vocab_size)
        2. context 见过、token 没见过：
           无平滑  0
           laplace 1 / (total + vocab_size)
        3. context 整个没见过：
           无平滑  0
           laplace 1 / vocab_size      ← 退化为均匀分布

    提示：model.get(context) 在 context 不存在时返回 None，可以据此分情况。
    参考规模：约 10 行。
    """
    counter = model.get(context)
    if counter == None:
        if smoothing == None:return 0
        return 1.0 / vocab_size
    total = sum(counter.values())
    if token in counter:
        count = counter[token]
        if smoothing == None:return count / total
        return (count + 1) / (total + vocab_size)
    else:
        if smoothing == None: return 0
        return 1 / (total + vocab_size)
   
def calculate_perplexity(model, test_tokens, n, vocab_size, smoothing=None):
    """计算模型在测试集上的困惑度。

    步骤提示：
        1. 对 test_tokens 做与训练时一致的补全（开头 n-1 个 START，结尾 1 个 END）。
        2. 遍历补全后的序列：对每个位置，取出 context 和真实后继 token，
           调用你写的 token_probability 得到概率 p。
        3. 如果 p == 0：直接返回 float("inf")。
           不用觉得可惜——"无平滑模型在测试集上原地爆炸"本身就是
           最重要的实验现象，让它发生，并在课堂展示中讨论它。
           （加分项：顺便 print 一下零概率出现在第几个 token、context 是什么。）
        4. 否则累加 math.log(p)，最后返回 math.exp(-总对数概率 / M)，
           其中 M 是参与求和的 token 数（含 END，不含 START）。

    自检：用确定性语料（每个上下文只有一个后继）训练后，
    模型在训练语料上的无平滑困惑度应当恰好是 1.0。

    参考规模：约 12 行。
    """
    padded = [START] * (n - 1) + test_tokens + [END]
    log_sum = 0.0
    M = 0
    for i in range(len(padded) - n + 1):
        context = tuple(padded[i: i + n - 1])
        real_token = padded[i + n - 1]
        p = token_probability(model, context, real_token, vocab_size, smoothing)
        if p==0: 
            return float("inf")
        else:
            log_sum += math.log(p)
            M += 1
    return math.exp(-log_sum / M)
