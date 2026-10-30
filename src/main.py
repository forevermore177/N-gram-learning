"""main.py —— 命令行入口（完整提供）

用法（在项目根目录 24-NGram/ 下运行）：

    python src/main.py inspect  --n 2 --context 的     # 查看模型中某个上下文的统计
    python src/main.py inspect  --n 3                  # 查看模型整体统计
    python src/main.py generate --n 3 --length 200     # 生成一段文本
    python src/main.py compare                         # N=1..4 对比实验 + 困惑度表

公共参数：--data 指定语料（默认 data/hongloumeng.txt），--mode 选 char/word 分词。
"""
import argparse
import csv
import sys
from pathlib import Path

# 让 Windows 老式终端也能正常显示中文与符号
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from tokenizer import load_corpus, tokenize, detokenize, START, END
from model import build_ngram_model, describe_model
from generate import generate_text
from evaluate import split_train_test, build_vocab, calculate_perplexity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT_ROOT / "data" / "shakespeare.txt"


def cmd_inspect(args):
    tokens = load_corpus(args.data, args.mode)
    model = build_ngram_model(tokens, args.n)
    print(f"语料：{args.data}（{len(tokens)} 个 token，分词模式 {args.mode}）\n")

    if args.context is None:
        describe_model(model, args.n)
        return

    # 把用户输入的上下文字符串转成 n-1 元组
    ctx_tokens = tuple(args.context) if args.mode == "char" else tuple(args.context.split())
    if len(ctx_tokens) != args.n - 1:
        print(f"注意：n={args.n} 时上下文需要恰好 {args.n - 1} 个 token，"
              f"你给了 {len(ctx_tokens)} 个：{ctx_tokens}")
        return

    counter = model.get(ctx_tokens)
    if counter is None:
        print(f"上下文 {ctx_tokens} 在语料中从未出现——这就是“数据稀疏”。")
        return

    total = sum(counter.values())
    print(f"上下文 {ctx_tokens} 共出现 {total} 次，后继分布（前 10）：")
    for token, count in counter.most_common(10):
        bar = "█" * max(1, round(30 * count / total))
        print(f"  {token!r:<12} {count:>4} 次  {count / total:6.1%}  {bar}")


def cmd_generate(args):
    tokens = load_corpus(args.data, args.mode)
    model = build_ngram_model(tokens, args.n)
    result = generate_text(model, args.n, length=args.length, seed=args.seed,
                           top_k=args.top_k, temperature=args.temperature)
    print(f"--- n={args.n}，seed={args.seed}，语料={Path(args.data).name} ---")
    print(detokenize(result, args.mode))

def cmd_compare(args):
    tokens = load_corpus(args.data, args.mode)
    train_tokens, test_tokens = split_train_test(tokens, args.train_ratio)
    vocab = build_vocab(train_tokens)
    print(f"语料：{args.data}")
    print(f"训练集 {len(train_tokens)} token / 测试集 {len(test_tokens)} token / 词表大小 {len(vocab)}\n")

    rows = []
    for n in args.ns:
        model = build_ngram_model(train_tokens, n)
        
        # 测试集：无平滑、Laplace
        ppl_test_none = calculate_perplexity(model, test_tokens, n, len(vocab), smoothing=None)
        ppl_test_lap = calculate_perplexity(model, test_tokens, n, len(vocab), smoothing="laplace")
        
        # 训练集：无平滑、Laplace
        ppl_train_none = calculate_perplexity(model, train_tokens, n, len(vocab), smoothing=None)
        ppl_train_lap = calculate_perplexity(model, train_tokens, n, len(vocab), smoothing="laplace")
        
        sample = detokenize(generate_text(model, n, length=args.sample_length, seed=args.seed), args.mode)
        rows.append((n, ppl_test_none, ppl_test_lap, ppl_train_none, ppl_train_lap, sample))
        
        print(f"===== n={n} 生成样例（seed={args.seed}）=====")
        print(sample[:200] + ("……" if len(sample) > 200 else ""))
        print()

    def fmt(x):
        return "inf" if x == float("inf") else f"{x:.2f}"

    # 打印超级详细的表格
    print("===== 困惑度对比（训练集 vs 测试集）=====")
    print(f"{'n':>3} | {'训练集(无平滑)':>14} | {'训练集(Laplace)':>15} | {'测试集(无平滑)':>14} | {'测试集(Laplace)':>15}")
    print("-" * 75)
    for n, p_test_none, p_test_lap, p_train_none, p_train_lap, _ in rows:
        print(f"{n:>3} | {fmt(p_train_none):>14} | {p_train_lap:>15.2f} | {fmt(p_test_none):>14} | {p_test_lap:>15.2f}")

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "perplexity.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "train_none", "train_laplace", "test_none", "test_laplace"])
        for n, p_test_none, p_test_lap, p_train_none, p_train_lap, _ in rows:
            writer.writerow([n, p_train_none, p_train_lap, p_test_none, p_test_lap])
    print(f"\n困惑度数据已保存到 {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure()
        # 画四条线！完美呈现所有现象
        plt.plot([r[0] for r in rows], [r[3] for r in rows], "o--", label="Train (No Smoothing)", color="green", alpha=0.6)
        plt.plot([r[0] for r in rows], [r[4] for r in rows], "o-", label="Train (Laplace)", color="green")
        plt.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="Test (Laplace)", color="red")
        
        plt.xlabel("n")
        plt.ylabel("perplexity")
        plt.title("N vs Perplexity (Train vs Test)")
        plt.legend()
        fig_path = out_dir / "perplexity.png"
        plt.savefig(fig_path, dpi=150)
        print(f"曲线图已保存到 {fig_path}")
    except ImportError:
        print("（未检测到 matplotlib，跳过自动绘图；可直接用 Excel 打开 CSV 画图）")

def main():
    parser = argparse.ArgumentParser(description="N-gram 文本风格模仿生成器")

    # 公共参数放在父解析器里，这样每个子命令后面都能跟 --data / --mode
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", default=str(DEFAULT_DATA), help="语料文件路径（UTF-8 文本）")
    common.add_argument("--mode", default="char", choices=["char", "word"], help="分词模式")

    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", parents=[common], help="查看模型统计")
    p_inspect.add_argument("--n", type=int, default=2)
    p_inspect.add_argument("--context", default=None, help="要查询的上下文（char 模式直接写字，word 模式用空格分隔）")

    p_gen = sub.add_parser("generate", parents=[common], help="生成文本")
    p_gen.add_argument("--n", type=int, default=3)
    p_gen.add_argument("--length", type=int, default=200)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--top-k", dest="top_k", type=int, default=None, help="进阶部分扩展：Top-k 采样")
    p_gen.add_argument("--temperature", type=float, default=1.0, help="进阶部分扩展：温度采样")

    p_cmp = sub.add_parser("compare", parents=[common], help="N=1..4 生成与困惑度对比实验")
    p_cmp.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3, 4])
    p_cmp.add_argument("--sample-length", dest="sample_length", type=int, default=200)
    p_cmp.add_argument("--train-ratio", dest="train_ratio", type=float, default=0.8)
    p_cmp.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    handler = {"inspect": cmd_inspect, "generate": cmd_generate, "compare": cmd_compare}[args.command]
    try:
        handler(args)
    except NotImplementedError as e:
        # 学生还没完成对应任务时，给出友好提示而不是堆栈
        print(f"\n{e}")
        print("提示：先运行 python src/check.py 查看当前进度。")
        sys.exit(1)


if __name__ == "__main__":
    main()
