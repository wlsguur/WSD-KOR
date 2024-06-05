"""
    prompt를 생성하여 모델에 넣어줄 입력 형태로 변환

    prompt format:
        주어진 문장을 보고, 단어의 의미를 묻는 질문에 문맥을 고려하여 답하세요.
        문장: 문장.
        문장에서의 "타겟 단어"은(는) "후보 단어 의미"의 의미로 쓰였습니까?

"""
import json
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

def prompt_generator(contexts, dictionary, use_all=False, split=False, test_size = 0.3):
    """
        args:
            contexts =
            {
                'form': [form1, form2, ...], : list of str
                'WSD' = [ '[{'word': wsd1, 'sense_id': id1}, ...]', ... ], : list of str
            }
            dictionary =
            {
                '단어1': {'key': [단어1_품사1, 단어1_품사2, ...], 'definition': [뜻1, 뜻2, ...], 'sense_no': [1, 2, ...]}
                '단어2': {'key': [단어2_품사1, 단어2_품사2, ...], 'definition': [뜻1, 뜻2, ...], 'sense_no': [1, 2, ...]}
                ...
            }
    """
    contexts['WSD'] = contexts['WSD'].fillna('[]')
    contexts['WSD'] = contexts['WSD'].str.replace("'", '"')
    contexts['WSD'] = contexts['WSD'].apply(lambda x: json.loads(x)) # convert str to list of dict

    inputs = []
    labels = []

    for i in range(len(contexts)):
        context = contexts['form'][i]
        line = contexts['WSD'][i] # list of dict
        for d in line:
            target_wsd, target_sense_id = d['word'], d['sense_id']
            if target_sense_id not in dictionary[target_wsd]["sense_no"]:
                continue
            if len(dictionary[target_wsd]["definition"]) > 3:
                idx = dictionary[target_wsd]["sense_no"].index(target_sense_id)
                if idx >= 3:
                    idxs = [0, 1, idx]
                else:
                    idxs = range(3)
            else:
                idxs = range(len(dictionary[target_wsd]["definition"]))
            for j in idxs:
                candidate_definition = dictionary[target_wsd]["definition"][j]
                candidate_sense_id = dictionary[target_wsd]["sense_no"][j]
                input = (
                    f"주어진 문장을 보고, 단어의 의미를 묻는 질문에 문맥을 고려하여 답하세요.\n"
                    f"문장: {context}\n"
                    f"문장에서의 \"{target_wsd}\"은(는) \"{candidate_definition}\"의 의미로 쓰였습니까?\n"
                )
                inputs.append(input)
                labels.append(1 if candidate_sense_id == target_sense_id else 0)

    if split:
        if use_all:
            val_inputs, test_inputs, val_labels, test_labels = train_test_split(inputs, labels, test_size=0.5, random_state=42, stratify=labels)
            val_data = {"inputs":  val_inputs, "labels": val_labels}
            test_data = {"inputs": test_inputs, "labels": test_labels}

            return val_data, test_data
        
        else:
            train_inputs, tmp_inputs, train_labels, tmp_labels = train_test_split(inputs, labels, test_size=test_size, random_state=42, stratify=labels)
            val_inputs, test_inputs, val_labels, test_labels = train_test_split(tmp_inputs, tmp_labels, test_size=0.5, random_state=42, stratify=tmp_labels)

            train_data = {"inputs":  train_inputs, "labels": train_labels}
            val_data = {"inputs":  val_inputs, "labels": val_labels}
            test_data = {"inputs": test_inputs, "labels": test_labels}

            return train_data, val_data, test_data

    else:
        return {"inputs": inputs, "labels": labels}

class PromptDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.inputs = data["inputs"]
        self.labels = data["labels"]
        self.tokenizer = tokenizer
        self.max_length = max(len(tokenizer.encode(input_text)) for input_text in self.inputs)
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = self.inputs[idx]
        label = self.labels[idx]

        input = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = input['input_ids'][0]
        attention_mask = input['attention_mask'][0]

        return input_ids, attention_mask, label