import re

def parse_mmlu_file(file_path):
    """
    与えられた形式のテキストファイルを解析し、
    [Final Consensus Answer] answer: が A～D（N/A以外）の問題だけを対象に
    Total token count for debate: の数値を集計して、
    合計値と平均値を返す関数。
    """
    
    # ファイルの内容を全て読み込み
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 各問題ブロックごとに分割
    #   例: "--- MMLU Q1 ---" から始まるブロックを取り出す
    #      ただし先頭に空要素が来るかもしれないのでフィルタする
    problem_blocks = re.split(r'-{3,}\s*MMLU\s+Q\d+\s*-{3,}', text)
    problem_blocks = [block.strip() for block in problem_blocks if block.strip()]

    token_counts = []

    for block in problem_blocks:
        # [Final Consensus Answer] を探す
        #  例: [Final Consensus Answer] answer: A
        match_answer = re.search(r'\[Final Consensus Answer\]\s+answer:\s+(\S+)', block)
        if match_answer:
            final_answer = match_answer.group(1).strip()
            print(f"final_answer: {final_answer}")
            # A, B, C, D のいずれかなら対象とする
            if final_answer in ['A', 'B', 'C', 'D']:
                # "Total token count for debate: XXX" の部分を探す
                match_tokens = re.search(r'Total token count for debate:\s*(\d+)', block)
                if match_tokens:
                    token_count = int(match_tokens.group(1))
                    token_counts.append(token_count)

    # 集計
    if not token_counts:
        print("有効なデータが見つかりませんでした。")
        return

    total_tokens = sum(token_counts)
    average_tokens = total_tokens / len(token_counts)

    print(f"３択がA～D（N/A以外）だった問題数  : {len(token_counts)}")
    print(f"Total token countの合計値         : {total_tokens}")
    print(f"Total token countの平均値         : {average_tokens:.2f}")


if __name__ == "__main__":
    # 解析したいファイルのパスを指定
    file_path = "results/debate_log_TeamT2_20250212_105424.txt"
    
    parse_mmlu_file(file_path)
