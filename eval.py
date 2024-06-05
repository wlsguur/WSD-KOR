from sklearn.metrics import classification_report
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from prompt import prompt_generator, PromptDataset
from utils import read_config, read_files
from baseline import Baseline

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Eval(Baseline):
    def __init__(self, config):
        super().__init__(config)
        self.use_all_corpus = config["use_all_corpus"]
        self.batch_size = config["batch_size"]
        self.run_name = config["run_name"]
        self.load_checkpoint = config["load_checkpoint"]
        self.checkpoint_load_path = config["checkpoint_load_path"]

        if self.load_checkpoint:
            state_dict = torch.load(self.checkpoint_load_path)
            self.model.load_state_dict(state_dict)
            print(f"Checkpoint loaded from {self.checkpoint_load_path}")

    def evaluation(self):
        _, val_contexts, dictionary = read_files(self.config)
        if self.use_all_corpus:
            _, test_data = prompt_generator(val_contexts, dictionary, use_all=True, split=True, test_size=0.5) # length: 374,927 / 2
        else:
            # length: 56,240
            _, _, test_data = prompt_generator(val_contexts, dictionary, use_all=False, split=True, test_size=0.3)
        test_dataset = PromptDataset(test_data, tokenizer=self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        total_preds = []
        total_labels = []
        device = self.device
        self.model.eval()
        self.model.to(device)

        with torch.no_grad():
            for input_ids, attention_masks, labels in tqdm(test_loader, desc="Evaluation ..."):
                input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)[0]
                preds = torch.argmax(outputs, dim=1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())

        print(f"Evaluation result:")
        report = classification_report(y_true=total_labels, y_pred=total_preds, digits=4)
        print(report)

        output_dir = os.path.join(self.run_name)
        if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)

if __name__ == "__main__":
    config = read_config()
    eval = Eval(config)
    eval.evaluation()