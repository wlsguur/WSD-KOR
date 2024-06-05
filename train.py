from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import yaml
import wandb

from prompt import prompt_generator, PromptDataset
from utils import read_config, read_files
from baseline import Baseline

class Train(Baseline):
    def __init__(self, config):
        super().__init__(config)
        self.batch_size = config["batch_size"]
        self.num_epochs = config["num_epochs"]
        self.lr = config["lr"]
        self.use_all_corpus = config["use_all_corpus"]
        self.run_name = config["run_name"]
        self.load_checkpoint = config["load_checkpoint"]
        self.checkpoint_load_path = config["checkpoint_load_path"]
        self.use_wandb = config["use_wandb"]
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.optimizer = None
        self.criterion = None
        self.prompt_format = None

        if self.load_checkpoint:
            state_dict = torch.load(self.checkpoint_load_path)
            self.model.load_state_dict(state_dict)
            print(f"Checkpoint loaded from {self.checkpoint_load_path}")

    def ready_for_train(self):
        print("Reading files ...")
        train_contexts, val_contexts, dictionary = read_files(self.config)

        print("Generating prompt ...")
        if self.use_all_corpus:
            train_data = prompt_generator(train_contexts, dictionary, use_all=True, split=False)
            # length: 3,390,121
            val_data, test_data = prompt_generator(val_contexts, dictionary, use_all=True, split=True, test_size=0.5)
            # length: 374,927 / 2 each
        else:
            train_data, val_data, test_data = prompt_generator(val_contexts, dictionary, use_all=False, split=True, test_size=0.3)
            # length: 262,448 | 56,239 | 56,240
        self.prompt_format = test_data["inputs"][0]

        print("Making datasets ...")
        train_dataset = PromptDataset(train_data, tokenizer=self.tokenizer)
        val_dataset = PromptDataset(val_data, tokenizer=self.tokenizer)
        test_dataset = PromptDataset(test_data, tokenizer=self.tokenizer)

        print("Converting to dataloader ...")
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)

        self.optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        print("Ready for training !")

        print(f"Train dataloader length: {len(self.train_loader)}")
        print(f"Validation dataloader length: {len(self.val_loader)}")
        print(f"Test dataloader length: {len(self.test_loader)}")

    def train(self):
        device = self.device
        print(f"Using Device: {device}")

        self.model.to(device)

        checkpoint_dir = os.path.join(self.run_name, f"checkpoints")
        if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)

        config_path = os.path.join(self.run_name, f"config.yaml")
        with open(config_path, 'w', encoding="utf-8") as f:
            self.config["prompt_format"] = self.prompt_format
            yaml.dump(self.config, f, allow_unicode=True)
        print(f"Config saved at {config_path}")

        num_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            self.model.train()
            for step, (input_ids, attention_masks, labels) in tqdm(enumerate(self.train_loader), desc=f"Epoch {epoch+1} train"):
                input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)[0]
                loss = self.criterion(outputs, labels)
                if step % 50 == 0:
                    if self.use_wandb == True:
                        wandb.log({"train_loss": loss.item()}, step= epoch*num_steps + step)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            torch.save(self.model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

            self.model.eval()
            total_preds = []
            total_labels = []
            with torch.no_grad():
                for input_ids, attention_masks, labels in tqdm(self.val_loader, desc=f"Epoch {epoch+1} validation"):
                    input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)[0]
                    preds = torch.argmax(outputs, dim=1)
                    total_preds.extend(preds.cpu().numpy())
                    total_labels.extend(labels.cpu().numpy())

            print(f"Epoch {epoch+1} validation result:")
            print(classification_report(y_true=total_labels, y_pred=total_preds, digits=4))

            acc = accuracy_score(y_true=total_labels, y_pred=total_preds)
            f1_weighted = f1_score(y_true=total_labels, y_pred=total_preds, average="weighted")
            if self.use_wandb == True:
                wandb.log({"val_acc": acc, "val_f1_weigthed": f1_weighted})

    def evaluation(self):
        device = self.device
        self.model.eval()
        self.model.to(device)
        total_preds = []
        total_labels = []
        with torch.no_grad():
            for input_ids, attention_masks, labels in tqdm(self.test_loader, desc=f"Evalutation ..."):
                input_ids, attention_masks, labels = input_ids.to(device), attention_masks.to(device), labels.to(device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_masks)[0]
                preds = torch.argmax(outputs, dim=1)
                total_preds.extend(preds.cpu().numpy())
                total_labels.extend(labels.cpu().numpy())

        print(f"Evaluation result:")
        report = classification_report(y_true=total_labels, y_pred=total_preds, digits=4)
        print(report)

        acc = accuracy_score(y_true=total_labels, y_pred=total_preds)
        f1_weighted = f1_score(y_true=total_labels, y_pred=total_preds, average="weighted")
        if self.use_wandb == True:
            wandb.log({"test_acc": acc, "test_f1_weigthed": f1_weighted})

        output_dir = os.path.join(self.run_name)
        if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

        report_path = os.path.join(output_dir, "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Evaluation result saved at {report_path}")
    
if __name__ == "__main__":
    config = read_config()
    train = Train(config)

    if train.use_wandb == True:
        wandb.init(project="NLP-WSD-KOR", config=config)
        wandb.config["prompt_format"] = train.prompt_format
        wandb.run.name = config["run_name"]
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("val_acc", summary="max")
        wandb.define_metric("val_f1_weighted", summary="max")

    train.ready_for_train()
    train.train()
    train.evaluation()
    
    if train.use_wandb == True:
        wandb.finish()