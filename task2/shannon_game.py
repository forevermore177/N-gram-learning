import csv
import sys
import math
from pathlib import Path

# 让终端能正常显示中文
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def load_text():
    """自动读取同目录下的 renewal.txt，并清理换行和多余空格"""
    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / "renewal.txt"
    
    if not file_path.exists():
        print(f"❌ 找不到文件：{file_path}")
        print("请确保 renewal.txt 和本脚本放在同一个 task2 文件夹下！")
        sys.exit(1)
        
    raw_text = file_path.read_text(encoding="utf-8")
    # 去掉所有换行、制表符、空格（香农实验通常只测字符，空格/换行会干扰测试）
    clean_text = "".join(raw_text.split())
    
    print(f"📄 成功加载文本！共 {len(clean_text)} 个字。")
    return clean_text

def run_game():
    target_text = load_text()
    
    print("=" * 55)
    print("🎮 欢迎来到香农猜字游戏 (Shannon Guessing Game)")
    print("规则：屏幕会显示已经猜出的前文，你需要输入下一个字。")
    print("     猜对自动跳下一个，猜错请继续。")
    print("     ⌨️  输入 'skip' 直接跳过（算作猜 20 次）。")
    print("     ⌨️  输入 'quit' 提前结束游戏。")
    print("=" * 55)

    guesses_data = []  # 记录每个字的猜测次数
    total_chars = len(target_text)
    
    for i in range(total_chars):
        # 为了屏幕不占太满，只显示最近 15 个字
        context = target_text[max(0, i-15):i]
        true_char = target_text[i]
        
        tries = 0
        while True:
            display_context = context if i > 0 else "（这是第一个字）"
            prompt = f"\n进度 [{i+1}/{total_chars}] 前文：...{display_context}\n👉 请猜下一个字："
            
            try:
                user_input = input(prompt).strip()
            except KeyboardInterrupt:
                print("\n👋 强行退出游戏。")
                sys.exit(0)
            
            if user_input.lower() == 'quit':
                print("\n🛑 提前结束游戏，正在统计已完成的部分...")
                break
                
            if user_input.lower() == 'skip':
                tries = 20
                print(f"⏭️ 跳过。正确字是：'{true_char}'")
                break
                
            if not user_input:
                print("⚠️ 请输入一个字符（或 skip/quit）。")
                continue
                
            tries += 1
            
            if user_input == true_char:
                print(f"✅ 猜对了！用了 {tries} 次。")
                break
            else:
                print(f"❌ 不对，再试试...")

        # 如果中途 quit，保存已经猜的部分并退出循环
        if user_input.lower() == 'quit':
            break
            
        guesses_data.append(tries)

    # ================= 结算统计 =================
    print("\n" + "=" * 55)
    print("🎉 游戏结束！统计结果：")
    
    if not guesses_data:
        print("没有完成任何猜测，无法统计。")
        return
        
    finished_chars = len(guesses_data)
    total_guesses = sum(guesses_data)
    avg_guesses = total_guesses / finished_chars
    first_try = sum(1 for g in guesses_data if g == 1)
    
    # 计算香农的近似信息量 (以 2 为底的对数)
    bits_per_char = math.log2(avg_guesses) if avg_guesses > 0 else 0
    
    print(f"实际测试字数：{finished_chars}")
    print(f"总猜测次数：{total_guesses}")
    print(f"平均每个字猜 {avg_guesses:.2f} 次")
    print(f"第一次就猜中的字数：{first_try} 个（占 {first_try/finished_chars:.1%}）")
    print(f"💡 人类在此文本上的近似信息量： {bits_per_char:.2f} bits/字")
    print("=" * 55)

    # ================= 保存 CSV =================
    current_dir = Path(__file__).resolve().parent
    csv_filename = current_dir / "shannon_guesses.csv"
    
    with open(csv_filename, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["位置", "真实字", "猜测次数"])
        for idx, (char, g) in enumerate(zip(target_text[:finished_chars], guesses_data), 1):
            writer.writerow([idx, char, g])
    
    print(f"📁 详细数据已保存到 {csv_filename}")
    print("接下来，请用你的 N=2 和 N=3 模型对这段文本算 PPL，并取 log2 对比！")

if __name__ == "__main__":
    run_game()