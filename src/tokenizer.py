"""tokenizer.py —— 把原始文本切成 token 序列（完整提供，可直接使用）

token 是什么取决于你怎么切：
  - mode="char"：每个汉字/标点/英文字母都是一个 token（推荐用于中文，无需任何分词库）
  - mode="word"：按单词切（推荐用于英文，简单的正则分词）

如果你想改进分词（比如中文按词切），改这里就好——这正是进阶部分的一个入口。
"""
import re
from pathlib import Path

# 特殊 token：保证不会与正常文本冲突
START = "<START>"
END = "<END>"


def tokenize(text, mode="char"):
    """把字符串切成 token 列表。"""
    if mode == "char":
        # 去掉所有空白（空格、换行），剩下的每个字符都是一个 token
        return [ch for ch in text if not ch.isspace()]
    elif mode == "word":
        # 简单英文分词：小写后提取字母串（含所有格如 don't）
        return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    else:
        raise ValueError(f"未知分词模式: {mode!r}（目前支持 'char' 和 'word'）")


def detokenize(tokens, mode="char"):
    """把 token 列表拼回可读的字符串。特殊 token（<START>/<END>）会被跳过。"""
    tokens = [t for t in tokens if t not in (START, END)]
    if mode == "char":
        return "".join(tokens)
    return " ".join(tokens)


def load_corpus(path, mode="char"):
    """读取 UTF-8 文本文件并分词，返回 token 列表。"""
    text = Path(path).read_text(encoding="utf-8")
    return tokenize(text, mode)
