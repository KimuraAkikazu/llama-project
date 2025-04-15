import pickle
import os

class dataloader:
    FILE_PATH = {
        "math": "./eval_data/math.pkl",
        "chess": "./eval_data/chess.pkl",
        "mmlu": "./eval_data/mmlu.pkl",
    }
    def __init__(self, name: str, n_case: int = 50):
        assert name.lower() in ["math", "chess", "mmlu"], f"dataset {name} is not valid."
        self.dataset = name.lower()
        self.n_case = n_case
        self.database = self._load_dataset()
        self.mode = "question"
    def _load_dataset(self):
        fpath = self.FILE_PATH[self.dataset]
        print("data_path:", fpath)
        with open(fpath, "rb") as f:
            db = pickle.load(f)
        return db
    def set_mode(self, mode: str):
        """
        mode=question -> 各問題 (Q,A,B,C,D) のタプル
        mode=answer   -> 正解のみ
        mode=all      -> {"task_info": (Q,A,B,C,D), "answer": X}
        """
        assert mode in ["all", "question", "answer"], f"mode {mode} not valid."
        self.mode = mode
    def __len__(self):
        real_count = len(self.database["task_info"])
        return min(self.n_case, real_count)
    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("dataloader out of range")
        if self.mode == "question":
            return self.database["task_info"][idx]
        elif self.mode == "answer":
            return self.database["answer"][idx]
        else:
            return {
                "task_info": self.database["task_info"][idx],
                "answer": self.database["answer"][idx]
            }
