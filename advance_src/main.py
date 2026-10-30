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

def sample_segments(tokens, k, length):
    """从 tokens 中等距截取 k 段，每段约 length 个 token（覆盖整个序列）。"""
    if len(tokens) < k * length:
        length = max(1, len(tokens) // k)   # 语料太短时自动缩短每段
    step = max(1, len(tokens) // k)
    segs = []
    for i in range(k):
        start = i * step
        seg = tokens[start:start + length]
        if seg:
            segs.append(seg)
    return segs

def run_compare_for(path, args, label=None):
    """对一份语料跑完整 compare：生成样例 + 四列困惑度表 + CSV + 曲线图。"""
    tokens = load_corpus(path, args.mode)
    train_tokens, test_tokens = split_train_test(tokens, tokens, args.train_ratio)
    vocab = build_vocab(train_tokens)
    tag = label or Path(path).stem
    print(f"语料：{path}（{tag}）")
    print(f"训练集 {len(train_tokens)} token / 测试集 {len(test_tokens)} token / 词表大小 {len(vocab)}\n")

    rows = []
    for n in args.ns:
        model = build_ngram_model(train_tokens, n)
        ppl_test_none = calculate_perplexity(model, test_tokens, n, len(vocab), smoothing=None)
        ppl_test_lap  = calculate_perplexity(model, test_tokens, n, len(vocab), smoothing="laplace")
        ppl_train_none = calculate_perplexity(model, train_tokens, n, len(vocab), smoothing=None)
        ppl_train_lap  = calculate_perplexity(model, train_tokens, n, len(vocab), smoothing="laplace")
        sample = detokenize(generate_text(model, n, length=args.sample_length, seed=args.seed), args.mode)
        rows.append((n, ppl_test_none, ppl_test_lap, ppl_train_none, ppl_train_lap, sample))

        print(f"===== {tag} | n={n} 生成样例（seed={args.seed}）=====")
        print(sample[:200] + ("……" if len(sample) > 200 else ""))
        print()

    def fmt(x):
        return "inf" if x == float("inf") else f"{x:.2f}"

    print(f"===== {tag} 困惑度对比（训练集 vs 测试集）=====")
    print(f"{'n':>3} | {'训练集(无平滑)':>14} | {'训练集(Laplace)':>15} | {'测试集(无平滑)':>14} | {'测试集(Laplace)':>15}")
    print("-" * 75)
    for n, p_test_none, p_test_lap, p_train_none, p_train_lap, _ in rows:
        print(f"{n:>3} | {fmt(p_train_none):>14} | {p_train_lap:>15.2f} | {fmt(p_test_none):>14} | {p_test_lap:>15.2f}")
    print()

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = "" if label is None else f"_{tag}"       # 单语料保持原文件名
    csv_path = out_dir / f"perplexity{suffix}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["n", "train_none", "train_laplace", "test_none", "test_laplace"])
        for n, p_test_none, p_test_lap, p_train_none, p_train_lap, _ in rows:
            writer.writerow([n, p_train_none, p_train_lap, p_test_none, p_test_lap])
    print(f"困惑度数据已保存到 {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot([r[0] for r in rows], [r[3] for r in rows], "o--", label="Train (No Smoothing)", color="green", alpha=0.6)
        plt.plot([r[0] for r in rows], [r[4] for r in rows], "o-",  label="Train (Laplace)",      color="green")
        plt.plot([r[0] for r in rows], [r[2] for r in rows], "o-",  label="Test (Laplace)",       color="red")
        plt.xlabel("n")
        plt.ylabel("perplexity")
        plt.title(f"N vs Perplexity ({tag})")
        plt.legend()
        fig_path = out_dir / f"perplexity{suffix}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"曲线图已保存到 {fig_path}")
    except ImportError:
        print("（未检测到 matplotlib，跳过自动绘图；可直接用 Excel 打开 CSV 画图）")
    print()

def cmd_compare(args):
    if args.data_b:
        import random
        tokens_A = load_corpus(args.data, args.mode)
        tokens_B = load_corpus(args.data_b, args.mode)

        # 1) 各切 80/20，词表各建各的
        train_A, test_A = split_train_test(tokens_A, tokens_A, args.train_ratio)
        train_B, test_B = split_train_test(tokens_B, tokens_B, args.train_ratio)
        vocab_A = build_vocab(train_A)
        vocab_B = build_vocab(train_B)
        V_union = len(vocab_A | vocab_B)       # 所有 N 共用同一个 |V|，Laplace 公平

        a_name, b_name = Path(args.data).stem, Path(args.data_b).stem
        print(f"A = {a_name}: train {len(train_A)} / test {len(test_A)} / |V_A|={len(vocab_A)}")
        print(f"B = {b_name}: train {len(train_B)} / test {len(test_B)} / |V_B|={len(vocab_B)}")
        print(f"Laplace 平滑，统一 |V| = |V_A ∪ V_B| = {V_union}\n")

        # 2) 截 10 段（与 N 无关，只做一次），打乱
        segs_A = sample_segments(test_A, 5, 200)
        segs_B = sample_segments(test_B, 5, 200)
        assert len(segs_A) == 5 and len(segs_B) == 5, \
            f"段数不足：A={len(segs_A)}, B={len(segs_B)}（语料太短，减小每段长度）"
        samples = [(s, "A", a_name) for s in segs_A] + [(s, "B", b_name) for s in segs_B]
        random.seed(args.seed)
        random.shuffle(samples)

        # 3) 对每个 n 循环
        all_results = []     # (n, correct, total, acc, detail_rows)
        for n in args.ns:
            model_A = build_ngram_model(train_A, n)
            model_B = build_ngram_model(train_B, n)

            print(f"===== N={n} 作者归属测试（10 段 × 约200 token） =====")
            print(f"{'#':>3}  {'真实':>4}  {'预测':>4}  {'PPL_A':>10}  {'PPL_B':>10}  {'判定':>4}")
            print("-" * 52)

            correct = 0
            detail = []
            for i, (seg, true_label, true_name) in enumerate(samples, 1):
                ppl_A = calculate_perplexity(model_A, seg, n, V_union, smoothing="laplace")
                ppl_B = calculate_perplexity(model_B, seg, n, V_union, smoothing="laplace")
                pred = "A" if ppl_A < ppl_B else "B"
                ok = (pred == true_label)
                correct += ok
                detail.append((i, true_label, true_name, pred, ppl_A, ppl_B, ok))
                mark = "✓" if ok else "✗"
                print(f"{i:>3}  {true_label:>4}  {pred:>4}  {ppl_A:>10.2f}  {ppl_B:>10.2f}  {mark:>4}")

            acc = correct / len(samples)
            print(f"→ N={n} 准确率：{correct}/{len(samples)} = {acc:.1%}\n")
            all_results.append((n, correct, len(samples), acc, detail))

        # 4) 汇总表：N vs 鉴别准确率
        print("===== N vs 鉴别准确率 =====")
        print(f"{'N':>3} | {'正确':>8} | {'准确率':>8}")
        print("-" * 30)
        for n, correct, total, acc, _ in all_results:
            print(f"{n:>3} | {correct:>3}/{total:<4} | {acc:>7.1%}")
        print()

        # 5) 记录：明细 + 汇总
        out_dir = PROJECT_ROOT / "results"
        out_dir.mkdir(exist_ok=True)

        detail_path = out_dir / "attribution_detail_by_n.csv"
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "#", "true_label", "true_name", "pred_label",
                             "ppl_model_A", "ppl_model_B", "correct"])
            for n, _, _, _, detail in all_results:
                for r in detail:
                    writer.writerow([n, r[0], r[1], r[2], r[3],
                                     f"{r[4]:.2f}", f"{r[5]:.2f}", int(r[6])])
        print(f"逐段明细已保存到 {detail_path}")

        summary_path = out_dir / "attribution_by_n.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["n", "correct", "total", "accuracy", "V_union"])
            for n, correct, total, acc, _ in all_results:
                writer.writerow([n, correct, total, f"{acc:.3f}", V_union])
        print(f"汇总表已保存到 {summary_path}")

        # 6) 折线图：N vs 准确率
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            ns   = [r[0] for r in all_results]
            accs = [r[3] * 100 for r in all_results]
            plt.figure()
            plt.plot(ns, accs, "o-", color="#4C72B0", linewidth=2, markersize=8)
            for nn, aa in zip(ns, accs):
                plt.text(nn, aa, f"{aa:.0f}%", ha="center", va="bottom")
            plt.axhline(50, color="red", linestyle="--", alpha=0.5, label="Random baseline (50%)")
            plt.xticks(ns)
            plt.ylim(0, 105)
            plt.xlabel("n")
            plt.ylabel("attribution accuracy (%)")
            plt.title("N vs Attribution Accuracy")
            plt.legend()
            plt.grid(alpha=0.3)
            fig_path = out_dir / "attribution_by_n.png"
            plt.savefig(fig_path, dpi=150)
            plt.close()
            print(f"曲线图已保存到 {fig_path}")
        except ImportError:
            print("（未检测到 matplotlib，跳过绘图）")

        return

    run_compare_for(args.data, args)
    
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
    p_cmp.add_argument("--data-b", dest="data_b", default=None,
                   help="第二份语料；给出时做 10 段作者归属测试（A/B 各 5 段，PPL 低者判为作者）")
    p_cmp.add_argument("--cross-n", dest="cross_n", type=int, default=3,
                   help="交叉评估的 n（默认 3，即任务要求的 N=3）")

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
