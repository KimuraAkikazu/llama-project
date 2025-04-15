import re
import sys
import contextlib
from llama_cpp import Llama
import json
from enum import Enum

class OutputFormat(Enum):
    JSON = "json"
    PLAIN = "plain"

@contextlib.contextmanager
def suppress_stdout_stderr():
    with open('/dev/null', 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def filter_repetitions(text, max_length=200):
    """
    長すぎる行や「response is」などの冗長なメタ情報を除去する。
    """
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if len(line) > max_length and "response is" in line:
            continue
        filtered.append(line)
    return "\n".join(filtered)

class LlamaAgent:
    """
    各エージェントは、システムプロンプトにより性格特性を与え、
    議論中にその個性を反映した理由付け（reasoning）を生成する。
    """
    def __init__(self, name, personality_text, model, max_tokens=512):
        self.name = name
        self.personality_text = personality_text
        self.model = model
        self.max_tokens = max_tokens
        if personality_text == "":
            self.system_message = f"System: You are {self.name}.\n"
        else:
            # 性格特性に加え、議論中に個性（冷静さ、熱狂性、几帳面さなど）を反映する指示を追加
            self.system_message = (
                f"System: You are {self.name} with the following personality traits:\n{self.personality_text}\n"
                "Answer concisely and strictly adhere to these traits. "
                "Also, please ensure that your reasoning reflects your personality. For example, if you are calm, use measured and logical explanations; "
                "if you are enthusiastic, include personal insights. "
                "When answering multiple-choice questions, output your answer in JSON format with the keys \"reasoning\" and \"answer\"."
            )
        self.conversation_history = [self.system_message]

    def _trim_conversation_history(self, max_lines=6):
        # 常にシステムプロンプト（最初の1行）は保持し、直近(max_lines-1)行だけ残す
        if len(self.conversation_history) > max_lines:
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-(max_lines-1):]

    def reset_history(self):
        self.conversation_history = [self.system_message]
    
    def generate_response(self, prompt):
        # ユーザ発言を履歴に追加（文字列）
        self.conversation_history.append(f"User: {prompt}")
        self._trim_conversation_history(max_lines=10)
        
        # conversation_history は文字列のリストなので、各行を分解してメッセージ辞書のリストに変換する
        messages = []
        for line in self.conversation_history:
            # 既に辞書型の場合はそのまま利用（万が一）
            if isinstance(line, dict):
                messages.append(line)
                continue
            # 文字列の場合は、先頭の "Role:" 部分を切り出して辞書に変換
            if line.startswith("System:"):
                messages.append({"role": "system", "content": line[len("System:"):].strip()})
            elif line.startswith("User:"):
                messages.append({"role": "user", "content": line[len("User:"):].strip()})
            elif line.startswith(f"{self.name}:"):
                messages.append({"role": "assistant", "content": line[len(f"{self.name}:"):].strip()})
            else:
                # そのほかの行はユーザ発言として扱う
                messages.append({"role": "user", "content": line.strip()})
        
        with suppress_stdout_stderr():
            output = self.model.create_chat_completion(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                stop=["System:", "User:"],
                seed=-1
            )
        
        # llama-cpp-python の返り値は "choices" 内の "message" キーに回答内容が入っている前提
        try:
            json_response = json.loads(output['choices'][0]['message']['content'].strip())
        except json.JSONDecodeError:
            json_response = {"reasoning": "", "answer": output['choices'][0]['message']['content'].strip()}
        
        # 生成結果を会話履歴に追加
        self.conversation_history.append(f"{self.name}: {output['choices'][0]['message']['content'].strip()}")
        return json_response



    def get_bfi_score(self, question_text: str, question_index: int, n_questions: int) -> int:
        system_prompt = (
            f"System: You are {self.name} with personality traits:\n{self.personality_text}\n"
            "Here are a number of characteristics that may or may not apply to you. For example, do you agree that you are someone who likes to spend time with others? Please write a number next to each statement to indicate the extent to which you agree or disagree with that statement.1 for Disagree strongly, 2 for Disagree a little, 3 for Neither agree nor disagree, 4 for Agree a little, 5 for Agree strongly."
            "For the following BFI question, respond with ONLY a single digit (1-5) without explanation."
        )
        user_prompt = f"BFI Q{question_index}/{n_questions}: {question_text}\n(1-5)?"
        full_prompt = f"{system_prompt}\nUser: {user_prompt}\n{self.name}:"
        stop_tokens = ["Agent1:", "Agent2:", "Agent3:", "System:", "User:", "\n\n"]
        with suppress_stdout_stderr():
            output = self.model(
                full_prompt,
                max_tokens=20,
                temperature=0.7,
                top_p=0.9,
                stop=stop_tokens,
                seed=-1
            )
        if 'choices' in output and len(output['choices']) > 0:
            raw = output['choices'][0]['text'].strip()
            m = re.search(r"\b([1-5])\b", raw)
            print(f"{self.name} BFI response: {m.group(1) if m else 'N/A'}")
            if m:
                return int(m.group(1))
        return 0

class AgentTriad:
    """
    3体のエージェントによるディベートを管理するクラス。
    ターン1では初回回答、ターン2以降では他エージェントの回答を参照して更新回答を生成する。
    最終回答は、3ターン目の各エージェントのJSON出力から「answer」を抽出し、多数決で決定する。
    """
    def __init__(self, agentX, agentY, agentZ):
        self.agents = [agentX, agentY, agentZ]
        self.round_responses = {}

    def conduct_discussion(self, topic_prompt, max_turns=3):
        print("\n=== Round 1 ===")
        self.round_responses[1] = {}
        # ターン1：初回回答
        round1_prompt = (
            "Can you answer the following question as accurately as possible? "
            f"{{{topic_prompt}}} Explain your answer, putting the answer in the form (X) at the end of your response. "
            "Please also reflect your personality in your explanation."
        )
        for agent in self.agents:
            response = agent.generate_response(round1_prompt)
            self.round_responses[1][agent.name] = response

        # ターン2以降：他エージェントの直前の回答を参照するプロンプト
        for turn in range(2, max_turns+1):
            print(f"\n=== Round {turn} ===")
            self.round_responses[turn] = {}
            for agent in self.agents:
                other_responses = [self.round_responses[turn-1][a.name] for a in self.agents if a.name != agent.name]
                debate_prompt = (
                    "These are the solutions to the question from other agents:\n"
                    f"One agent solution: {other_responses[0]}\n"
                    f"Another agent solution: {other_responses[1]}\n\n"
                    "Based on your personality and the reasoning from the other agents, "
                    "please update your answer. Carefully review your own solution and that of the others, "
                    "and put your answer in the form (X) at the end of your response. "
                    "Output your answer in JSON format with the format: {\"reasoning\": \"\", \"answer\": \"\"}. "
                    "Please strictly output in JSON format."
                )
                response = agent.generate_response(debate_prompt)
                self.round_responses[turn][agent.name] = response

    def get_final_consensus(self):
        final_round = self.round_responses.get(max(self.round_responses.keys()), {})
        extracted = {}
        for agent_name, resp in final_round.items():
            # もし resp が dict 型なら、直接 "answer" キーを利用する
            if isinstance(resp, dict):
                ans_value = resp.get("answer", "")
                if isinstance(ans_value, int):
                    ans_value = str(ans_value)
                extracted[agent_name] = ans_value.strip() if ans_value else "N/A"
            else:
                # resp が文字列の場合は、正規表現で JSON 部分を抽出する
                m_json = re.search(r'(\{.*\})', resp, re.DOTALL)
                if m_json:
                    json_str = m_json.group(1)
                    try:
                        response_data = json.loads(json_str)
                        ans_value = response_data.get("answer", "")
                        if isinstance(ans_value, int):
                            ans_value = str(ans_value)
                        extracted[agent_name] = str(ans_value).strip('"') if ans_value else "N/A"
                    except json.JSONDecodeError:
                        extracted[agent_name] = "N/A"
                else:
                    m = re.search(r"\(([A-D])\)", resp)
                    if m:
                        extracted[agent_name] = m.group(1).upper()
                    else:
                        extracted[agent_name] = "N/A"
        votes = {}
        for ans in extracted.values():
            if ans != "N/A":
                votes[ans] = votes.get(ans, 0) + 1
        final_answer = max(votes, key=votes.get) if votes else "N/A"
        return final_answer
        
class ConsensusAgent:
    # 今回は多数決で最終回答を決定するため、使用しません。
    def __init__(self, name, model, max_tokens=128):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens

class BFIAnalyzerAgent:
    def __init__(self, model, max_tokens=64):
        self.model = model
        self.max_tokens = max_tokens

    def determine_score(self, question_text: str, persona_response: str) -> int:
        return 0
