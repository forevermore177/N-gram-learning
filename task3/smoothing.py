import math
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from tokenizer import load_corpus
from model import build_ngram_model
from evaluate import build_vocab, calculate_perplexity   # ← 新增 calculate_perplexity

def get_jm_prob(w, ctx3, ctx2, models, vocab_size, lam):
    p3_count = models[3].get(ctx3, {}).get(w, 0)
    p3_total = sum(models[3].get(ctx3, {}).values())
    p3 = (p3_count + 1) / (p3_total + vocab_size) if p3_total > 0 else 1.0 / vocab_size

    p2_count = models[2].get(ctx2, {}).get(w, 0)
    p2_total = sum(models[2].get(ctx2, {}).values())
    p2_ml = (p2_count + 1) / (p2_total + vocab_size) if p2_total > 0 else 1.0 / vocab_size

    p1_count = models[1].get((), {}).get(w, 0)
    p1_total = sum(models[1].get((), {}).values())
    p1_ml = (p1_count + 1) / (p1_total + vocab_size) if p1_total > 0 else 1.0 / vocab_size

    p2_interp = lam * p2_ml + (1 - lam) * p1_ml
    return lam * p3 + (1 - lam) * p2_interp

def calculate_jm_ppl(models, test_tokens, vocab_size, lam):
    total_log_prob = 0.0
    count = 0
    for i in range(2, len(test_tokens)):
        ctx3 = tuple(test_tokens[i-2:i])
        ctx2 = tuple(test_tokens[i-1:i])
        w = test_tokens[i]
        prob = get_jm_prob(w, ctx3, ctx2, models, vocab_size, lam)
        total_log_prob += math.log2(prob)
        count += 1
    return 2 ** (-total_log_prob / count)

def run_experiment():
    data_path = PROJECT_ROOT / "data" / "hongloumeng.txt"
    tokens = load_corpus(str(data_path), "char")
    split_idx = int(len(tokens) * 0.8)
    train_tokens, test_tokens = tokens[:split_idx], tokens[split_idx:]
    vocab = build_vocab(train_tokens)
    V = len(vocab)

    print("⏳ 正在训练 N=1,2,3 模型（全量语料）...")
    models = {1: build_ngram_model(train_tokens, 1),
              2: build_ngram_model(train_tokens, 2),
              3: build_ngram_model(train_tokens, 3)}

    # ===== 新增：全量 Laplace N=3 基线（口径与 main.py compare 一致）=====
    full_laplace_ppl = calculate_perplexity(models[3], test_tokens, 3, V, smoothing="laplace")
    print(f"\n📏 Full Laplace (N=3) PPL = {full_laplace_ppl:.2f}")

    # ===== 修改：扩展 λ 网格，往小值补，确认 half 最优是否还在边界 =====
    lambdas = [0.01, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.5, 0.7, 0.9]
    full_ppls = []

    print("\n📊 全量语料下的 Lambda vs PPL：")
    for lam in lambdas:
        ppl = calculate_jm_ppl(models, test_tokens, V, lam)
        full_ppls.append(ppl)
        print(f"λ = {lam:.2f} → PPL = {ppl:.2f}")

    print("\n✂️ 截取一半训练语料，重新训练...")
    half_train_tokens = train_tokens[:len(train_tokens)//2]
    models_half = {1: build_ngram_model(half_train_tokens, 1),
                   2: build_ngram_model(half_train_tokens, 2),
                   3: build_ngram_model(half_train_tokens, 3)}

    # ===== 新增：Half Laplace N=3 基线 =====
    half_laplace_ppl = calculate_perplexity(models_half[3], test_tokens, 3, V, smoothing="laplace")
    print(f"📏 Half Laplace (N=3) PPL = {half_laplace_ppl:.2f}")

    half_ppls = []
    print("\n📊 半量语料下的 Lambda vs PPL：")
    for lam in lambdas:
        ppl = calculate_jm_ppl(models_half, test_tokens, V, lam)
        half_ppls.append(ppl)
        print(f"λ = {lam:.2f} → PPL = {ppl:.2f}")

    # ===== 找最优点 + 打印降幅 =====
    best_idx = full_ppls.index(min(full_ppls))
    best_idx_half = half_ppls.index(min(half_ppls))
    print(f"\n✅ 全量最优 λ = {lambdas[best_idx]}, PPL = {full_ppls[best_idx]:.2f}")
    print(f"✅ 半量最优 λ = {lambdas[best_idx_half]}, PPL = {half_ppls[best_idx_half]:.2f}")

    print(f"\n📉 Full JM 相对 Laplace 降低: "
          f"{(full_laplace_ppl - full_ppls[best_idx]) / full_laplace_ppl:.2%}")
    print(f"📉 Half JM 相对 Laplace 降低: "
          f"{(half_laplace_ppl - half_ppls[best_idx_half]) / half_laplace_ppl:.2%}")

    # ================= 保存 CSV =================
    task3_dir = Path(__file__).resolve().parent
    csv_path = task3_dir / "jm_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lambda", "full_ppl", "half_ppl"])
        for lam, p_full, p_half in zip(lambdas, full_ppls, half_ppls):
            writer.writerow([lam, f"{p_full:.2f}", f"{p_half:.2f}"])
    print(f"\n📁 数据已保存到 {csv_path}")

    # ================= 画图 =================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 6))
        plt.plot(lambdas, full_ppls, "o-", label="Full Training Data", color="#4C72B0", linewidth=2)
        plt.plot(lambdas, half_ppls, "s--", label="Half Training Data", color="#DD8452", linewidth=2)

        # 两条 Laplace 基线
        plt.axhline(full_laplace_ppl, color="red", linestyle=":",
                    label=f"Full Laplace ({full_laplace_ppl:.2f})")
        plt.axhline(half_laplace_ppl, color="orange", linestyle=":",
                    label=f"Half Laplace ({half_laplace_ppl:.2f})")

        # 两个最优点
        plt.plot(lambdas[best_idx], full_ppls[best_idx], "o", markersize=10,
                 markerfacecolor="yellow", markeredgecolor="black", zorder=5)
        plt.annotate(f"best λ={lambdas[best_idx]}",
                     (lambdas[best_idx], full_ppls[best_idx]),
                     textcoords="offset points", xytext=(8, -18), fontsize=9)

        plt.plot(lambdas[best_idx_half], half_ppls[best_idx_half], "s", markersize=10,
                 markerfacecolor="yellow", markeredgecolor="black", zorder=5)
        plt.annotate(f"best λ={lambdas[best_idx_half]}",
                     (lambdas[best_idx_half], half_ppls[best_idx_half]),
                     textcoords="offset points", xytext=(8, -18), fontsize=9)

        plt.xlabel("Lambda (λ)")
        plt.ylabel("Perplexity (PPL)")
        plt.title("Jelinek-Mercer Smoothing: Lambda vs Perplexity")
        plt.legend()
        plt.grid(alpha=0.3)

        fig_path = task3_dir / "jm_curve.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"曲线图已保存到 {fig_path}")
    except ImportError:
        print("（未检测到 matplotlib，跳过绘图）")

if __name__ == "__main__":
    run_experiment()