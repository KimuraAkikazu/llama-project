import sys
import re
import csv
import os
from datetime import datetime
from llama_cpp import Llama
from agents import LlamaAgent, AgentTriad, ConsensusAgent, BFIAnalyzerAgent
from bfi import run_bfi_test_with_analyzer
from dataloader import dataloader
import config

def summarize_conversation(history, max_prompt_tokens=4000):
    if len(history) > 100:
        system_msg = history[0]
        rest = history[-100:]
        history = [system_msg] + rest
    return "\n".join(history)

def format_mmlu_question(question_tuple):
    question, opt1, opt2, opt3, opt4 = question_tuple
    return (
        f"Question: {question}\n"
        f"A. {opt1}\n"
        f"B. {opt2}\n"
        f"C. {opt3}\n"
        f"D. {opt4}"
    )

# 追加：簡易的なトークン数算出関数（whitespace分割）
# 既存の count_tokens 関数を修正
def count_tokens(text):
    # text が dict 型の場合は、"reasoning" と "answer" を結合する
    if isinstance(text, dict):
        combined_text = ""
        if "reasoning" in text:
            combined_text += text["reasoning"] + " "
        if "answer" in text:
            combined_text += text["answer"]
        text = combined_text.strip()
    return len(text.split())

def calculate_total_tokens(round_responses):
    total = 0
    for turn, responses in round_responses.items():
        for agent, resp in responses.items():
            total += count_tokens(resp)
    return total


if __name__ == "__main__":
    # 1. モデルのロード
    llama = Llama(
        model_path=config.get_model_path(),
        verbose=True,
        n_threads=8,
        n_gpu_layers=-1,
        chat_format="llama-3",
        n_ctx=8192
    )

    # 2. BigFive前提の性格特性辞書
    bigfive_prompts = {
        "AgentT1": (
            "You are a character with high Openness and high Agreeableness, paired with moderate Extraversion and moderate Conscientiousness. "
            "Your imaginative mind and warm, cooperative nature drive you to explore innovative ideas while nurturing harmonious interactions. "
            "You remain calm (low Neuroticism) even when facing challenges. "
            "Answer thoughtfully and creatively, ensuring your responses reflect empathy and originality."
        ),
        "AgentT2": (
            "You are a character with high Conscientiousness and high Extraversion, complemented by moderate Openness and low Agreeableness. "
            "Your decisive, organized, and assertive demeanor makes you a pragmatic leader who values efficiency and clarity. "
            "You maintain composure (low Neuroticism) and focus on delivering clear, goal-oriented responses without excessive sentiment. "
            "Answer in a direct and methodical manner, staying true to your results-driven mindset."
        ),
        "AgentT3": (
            "You are a character with high Openness and high Neuroticism, along with moderate levels of Conscientiousness, Extraversion, and Agreeableness. "
            "Your rich inner life fuels a deep creative insight, though it is often accompanied by intense emotional sensitivity and occasional self-doubt. "
            "Embrace your introspective and passionate nature; answer with nuanced, reflective responses that capture both your visionary ideas and your candid vulnerability."
        ),
        "AgentNone": (
            ""
        )
    }

    # 3. 実験するチーム構成を定義
    # それぞれの構成は (チーム名, [agent1_personality, agent2_personality, agent3_personality]) のタプルとする
    team_configurations = [
        ("TeamNone", [bigfive_prompts["AgentNone"], bigfive_prompts["AgentNone"], bigfive_prompts["AgentNone"]]),
        ("TeamMixed", [bigfive_prompts["AgentT1"], bigfive_prompts["AgentT2"], bigfive_prompts["AgentT3"]]),
        ("TeamT2", [bigfive_prompts["AgentT2"], bigfive_prompts["AgentT2"], bigfive_prompts["AgentT2"]])
    ]

    # それぞれのチーム構成ごとに実験を実施
    #for team_name, personalities in team_configurations: 
    team_name = "Teammixed"
    personalities = [bigfive_prompts["AgentT1"], bigfive_prompts["AgentT2"], bigfive_prompts["AgentT3"]]
    print(f"\n===== Running experiment for {team_name} =====\n")
    # エージェントの定義（各チームごとに名前は同じく Agent1, Agent2, Agent3 とする）
    agent1 = LlamaAgent("Agent1", personalities[0], llama, max_tokens=1024)
    agent2 = LlamaAgent("Agent2", personalities[1], llama, max_tokens=1024)
    agent3 = LlamaAgent("Agent3", personalities[2], llama, max_tokens=1024)
    all_persona_agents = [agent1, agent2, agent3]
    
    # 議論前BFIテスト（必要に応じて実施・保存）
    for ag in all_persona_agents:
            run_bfi_test_with_analyzer(ag, BFIAnalyzerAgent(llama), "Pre", f"bfi_results_pre_{team_name}_{datetime.now().strftime('%Y%m%d_%H')}.csv")
    
    # 4. MMLUデータセットの読み込み
    print("\n=== Loading tasks from MMLU dataset ===")
    dl = dataloader("mmlu", n_case=990)
    dl.set_mode("all")
    
    # 5. 3エージェントによるディベート
    from collections import defaultdict
    triad = AgentTriad(agent1, agent2, agent3)
    print(f"\n=== 3-Agent Debate on MMLU tasks for {team_name} ===")
    
    # チーム毎のディベートログファイルを実験日時付きで保存
    team_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debate_log_file = f"./results/debate_log_{team_name}_{team_timestamp}.txt"
    with open(debate_log_file, "w", encoding="utf-8") as log_f:
    
        results_csv = f"./results/mmlu_results_{team_name}_{team_timestamp}.csv"
        os.makedirs("./results", exist_ok=True)
        fieldnames = ["task_index", "question", "final_answer", "correct_answer", "is_correct", "token_count"]
    
        with open(results_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            num_correct = 0
            total = 0
    
            for idx in range(len(dl)):
                item = dl[idx]
                question_tuple = item["task_info"]
                correct_ans = item["answer"]
                question_text = format_mmlu_question(question_tuple)
                print(f"\n--- MMLU Q{idx+1} ---\n{question_text}\n")
                log_f.write(f"--- MMLU Q{idx+1} ---\n{question_text}\n")
    
                # 各エージェントの会話履歴をリセット
                for agent in all_persona_agents:
                    agent.reset_history()
    
                # ターン1：初回回答
                print("\n=== Round 1 ===")
                log_f.write("\n=== Round 1 ===\n")
                round1_prompt = (
                    "please answer the question with step-by-step reasoning. There is only one correct option. "
                    f"{question_text} "
                    "Output your answer in JSON format with the format: {\"reasoning\": \"\", \"answer\": \"\"}. Do not output any extra text."
                )
                triad.round_responses = {}
                triad.round_responses[1] = {}
                for agent in all_persona_agents:
                    resp = agent.generate_response(round1_prompt)
                    triad.round_responses[1][agent.name] = resp
                    print(f"{agent.name} (Turn 1): {resp}\n")
                    log_f.write(f"{agent.name} (Turn 1): {resp}\n")
    
                # ターン2とターン3
                for turn in range(2, 4):
                    print(f"\n=== Round {turn} ===")
                    log_f.write(f"\n=== Round {turn} ===\n")
                    triad.round_responses[turn] = {}
                    for agent in all_persona_agents:
                        other_resps = [triad.round_responses[turn-1][a.name] for a in all_persona_agents if a.name != agent.name]
                        debate_prompt = (
                            "These are the solutions to the question from other agents:\n"
                            f"One agent solution: {other_resps[0]}\n"
                            f"Another agent solution: {other_resps[1]}\n\n"
                            "Using the reasoning from the other agents as additional advice, can you give an updated answer? "
                            "Carefully review your own solution and that of the others."#and put your answer in the form (X) at the end of your response. "
                            "Output your answer in JSON format with the format: {\"reasoning\": \"\", \"answer\": \"\"}. Please strictly output in JSON format."
                        )
                        resp = agent.generate_response(debate_prompt)
                        triad.round_responses[turn][agent.name] = resp
                        print(f"{agent.name} (Turn {turn}): {resp}\n")
                        log_f.write(f"{agent.name} (Turn {turn}): {resp}\n")
    
                # 最終回答の多数決（3ターン目の各エージェントの回答から括弧内の値を抽出）
                final_answer = triad.get_final_consensus()
                print(f"\n[Final Consensus Answer] answer: {final_answer}\n")
                log_f.write(f"\n[Final Consensus Answer] answer: {final_answer}\n")
    
                # ディベート全体のトークン数を算出
                token_count = calculate_total_tokens(triad.round_responses)
                print(f"Total token count for debate: {token_count}")
                log_f.write(f"Total token count for debate: {token_count}\n")
    
                is_correct = (final_answer == correct_ans.upper()) if correct_ans else False
                writer.writerow({
                    "task_index": idx+1,
                    "question": question_tuple[0],
                    "final_answer": final_answer,
                    "correct_answer": correct_ans,
                    "is_correct": is_correct,
                    "token_count": token_count
                })
                if correct_ans:
                    total += 1
                    if is_correct:
                        num_correct += 1
    
            if total > 0:
                accuracy = num_correct / total
                print(f"\nOverall accuracy: {accuracy*100:.1f}% ({num_correct}/{total})")
                log_f.write(f"\nOverall accuracy: {accuracy*100:.1f}% ({num_correct}/{total})\n")
    
        print(f"\nResults saved: {results_csv}\n")
        log_f.write(f"\nResults saved: {results_csv}\n")
    
    # 6. 議論後BFIテストの実施（コメントアウト）
    # for ag in all_persona_agents:
    #     run_bfi_test_with_analyzer(ag, BFIAnalyzerAgent(llama), "Post", f"bfi_results_post_{ag.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
