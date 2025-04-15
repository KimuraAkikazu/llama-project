import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み
df = pd.read_csv("bfi_results_pre_TeamMixed_20250212_0953.csv")  # 適切なファイルパスに変更

# 数値データのみ抽出
bigfive_traits = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]

# エージェントごとにループしてプロット
agents = df["AgentName"].unique()
for agent in agents:
    agent_df = df[df["AgentName"] == agent]
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=agent_df[bigfive_traits])
    
    plt.title(f"Big Five Scores Distribution for {agent}")
    plt.xlabel("Big Five Traits")
    plt.ylabel("Score")
    plt.ylim(0, 50)  # スコア範囲の調整（適宜変更）

    # 画像として保存
    plt.savefig(f"{agent}_bigfive_boxplot.png", dpi=300)
    plt.close()  # メモリ節約のため閉じる
