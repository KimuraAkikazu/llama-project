# mbti_test.py

import csv
import json
import os
import re
from datetime import datetime
from enum import Enum

from agents import LlamaAgent, MBTIAnswer

# JSONファイルからMBTI質問を読み込む
with open("translated_mbti_ch2en.json", "r", encoding="utf-8") as f:
    MBTI_QUESTIONS = json.load(f)

TOTAL_QUESTIONS = len(MBTI_QUESTIONS)

def run_mbti_test(agent: LlamaAgent, test_phase: str, base_csv_file="results"):
    """
    translated_mbti_ch2en.jsonを参照し、MBTIの質問(1~93)に対して
    Llamaエージェントを使ってA/Bを答えさせる。
    回答を集計し、MBTIタイプを判定後、CSVに書き込む。
    """

    # MBTIモードON
    agent.set_mbti_mode(True)

    # A/Bの回答を保持
    answer_list = []

    for q_idx in range(1, TOTAL_QUESTIONS + 1):
        question_data = MBTI_QUESTIONS.get(str(q_idx))
        if not question_data:
            print(f"[WARNING] Question {q_idx} not found in JSON.")
            continue

        # プロンプトを作成
        user_prompt = (
            f"Question {q_idx}: {question_data['question_en']}\n"
            f"A. {question_data['A']}\n"
            f"B. {question_data['B']}\n"
            "Answer with only 'A' or 'B'."
        )

        chosen_option = None
        max_tries = 3

        for attempt in range(max_tries):
            response = agent.generate_response(user_prompt)
            print(f"[DEBUG] Agent {agent.name} Q{q_idx} Response: '{response}'")

            # 'A' または 'B' のみを抽出
            if response in [MBTIAnswer.A.value, MBTIAnswer.B.value]:
                chosen_option = response
                break
            else:
                # 再試行用の追加プロンプト
                user_prompt += "\nYour format was incorrect. Please respond with only 'A' or 'B'."

        if not chosen_option:
            print(f"[ERROR] {agent.name} failed Q{q_idx} after {max_tries} attempts. Test aborted.")
            agent.set_mbti_mode(False)
            return  # テストを中断

        answer_list.append(chosen_option)

    # MBTIモードOFF
    agent.set_mbti_mode(False)

    # 回答から軸を集計 (E/I, S/N, T/F, J/P)
    axis_count = {"E": 0, "I": 0, "S": 0, "N": 0, "T": 0, "F": 0, "J": 0, "P": 0}

    for q_idx, chosen_option in enumerate(answer_list, start=1):
        question_data = MBTI_QUESTIONS.get(str(q_idx))
        if not question_data:
            continue
        axis = question_data.get(chosen_option)
        if axis:
            axis_count[axis] += 1

    # 4軸それぞれで多いほうを選択
    def pick_axis(a, b):
        return a if axis_count[a] >= axis_count[b] else b

    final_mbti = ""
    final_mbti += pick_axis("E", "I")
    final_mbti += pick_axis("S", "N")
    final_mbti += pick_axis("T", "F")
    final_mbti += pick_axis("J", "P")

    # CSV出力準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(base_csv_file, exist_ok=True)
    csv_file = os.path.join(base_csv_file, f"mbti_results_{timestamp}.csv")

    fieldnames = ["AgentName", "TestPhase", "MBTI"]
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {
            "AgentName": agent.name,
            "TestPhase": test_phase,
            "MBTI": final_mbti
        }
        writer.writerow(row)

    print(f"[INFO] MBTI test done for {agent.name} (phase={test_phase}). Result={final_mbti}. Saved to {csv_file}")
