import json
import torch
from torch.utils.data import Dataset

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            problem_answer = json.loads(line)
            problem = problem_answer["question"]
            answer = problem_answer["answer"]
            answer_sentences = answer_split(answer)
            for i in range(len(answer_sentences)):
                step = answer_sentences[i]
                if len(step.strip()) == 0:
                    continue
                data_entity = {
                    "hints": problem + " " + "".join(answer_sentences[:i]),
                    "steps": answer_sentences[i],
                }
                data.append(data_entity)
    return data

def answer_split(answer):
    answer = answer.replace('\n\n', '\n')
    sentences = answer.split('\n')
    try:
        sentences = sentences[:-2] + [sentences[-2]+'\n'+sentences[-1]]
    except:
        sentences = answer.split('. ')
    return sentences

# Data Loader
class ProblemAnswerDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        """
        Args:
            file_path (str): Path to the dataset (JSONL file with {"problem": ..., "answer": ...}).
            tokenizer: A tokenizer (e.g., GPT tokenizer) for tokenizing input text.
            max_length (int): Maximum sequence length.
            eos_token_id (int): End-of-sequence token ID.
        """
        self.data = load_jsonl(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_entity = self.data[idx]
        hints = data_entity["hints"]
        steps = data_entity["steps"]
        # Tokenize
        hints_tokens = torch.tensor(self.tokenizer.encode(hints, max_length=256, truncation=True), dtype=torch.long)
        steps_tokens = torch.tensor(self.tokenizer.encode(steps, max_length=256, truncation=True), dtype=torch.long)
        return {
            "hints": hints,
            "steps": steps,
            "hints_tokens": hints_tokens,
            "steps_tokens": steps_tokens,
        }
        
        
class CollateFn:
    def __init__(self, eos_token_id, pad_token_id, sep_token_id):
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.sep_token_id = sep_token_id
    
    def __call__(self, batch):
        """Collate function for dynamic padding."""
        inputs_tokens = []
        hints = []
        steps = []
        sep_pos = []
        val_len = []
        for item in batch:
            hints.append(item["hints"])
            steps.append(item["steps"])
            hints_tokens = item["hints_tokens"]
            steps_tokens = item["steps_tokens"]
            input_tokens = torch.cat([hints_tokens, torch.tensor([self.sep_token_id], dtype=torch.long)])
            sep_pos.append(len(input_tokens))
            input_tokens = torch.cat([input_tokens, steps_tokens])
            # add eos_token_id
            input_tokens = torch.cat([input_tokens, torch.tensor([self.eos_token_id], dtype=torch.long)])
            inputs_tokens.append(input_tokens)
            val_len.append(len(input_tokens))
        max_input_len = max(len(input_tokens) for input_tokens in inputs_tokens) 
        # Padding
        input_ids = []
        attention_masks = []
        loss_masks = []
        hints_sep_ids = []
        hints_sep_attention_masks = []
        for i in range(len(inputs_tokens)):
            input_tokens = inputs_tokens[i]
            pad_len = max_input_len - len(input_tokens)
            input_id = torch.cat([input_tokens, torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            input_ids.append(input_id)
            attention_mask = torch.ones(len(input_tokens), dtype=torch.long)
            padded_attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            attention_masks.append(padded_attention_mask)
            loss_mask = torch.ones(len(input_tokens), dtype=torch.long)
            loss_mask[:sep_pos[i]] = 0
            padded_loss_mask = torch.cat([loss_mask, torch.zeros(pad_len, dtype=torch.long)])
            loss_masks.append(padded_loss_mask)
            hints_sep_ids.append(input_id[:sep_pos[i]])
        
        max_hints_sep_len = max(len(hints_sep_id) for hints_sep_id in hints_sep_ids)
        for i in range(len(hints_sep_ids)):
            pad_len = max_hints_sep_len - len(hints_sep_ids[i])
            hints_sep_attention_mask = torch.ones(len(hints_sep_ids[i]), dtype=torch.long)
            padded_hints_sep_attention_mask = torch.cat([hints_sep_attention_mask, torch.zeros(pad_len, dtype=torch.long)])
            hints_sep_attention_masks.append(padded_hints_sep_attention_mask)
            hints_sep_ids[i] = torch.cat([hints_sep_ids[i], torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
   
        return {
            "hints": hints,
            "steps": steps,
            "input_ids": torch.stack(input_ids),  # (batch, max_input_len)
            "hints_sep_ids": torch.stack(hints_sep_ids), # (batch, max_hints_sep_len)
            "hints_sep_attention_masks": torch.stack(hints_sep_attention_masks), # (batch, max_hints_sep_len)
            "attention_mask": torch.stack(attention_masks),  # (batch, max_input_len)
            "loss_mask": torch.stack(loss_masks),  # (batch, max_input_len)
            "sep_pos": sep_pos,
            "val_len": val_len,
        }