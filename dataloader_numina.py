import json
import torch
import re
from torch.utils.data import Dataset

def load_jsonl(file):
    data = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def split_solution_into_sentences(text: str):
    formula_pattern = re.compile(r'(\$\$.*?\$\$|\\\[.*?\\\])', re.DOTALL)
    enumerator_pattern = re.compile(r'(?:^|\s)\d+$')
    parts = formula_pattern.split(text)
    sentences = []
    logical_connectors = [
        "Therefore", "Thus", "Also", "Moreover",
        "Since", "Hence", "Then", "So", "Consequently"
    ]
    connector_pattern = re.compile(r'^\s*(?:' + '|'.join(logical_connectors) + r')\b', re.IGNORECASE)
    connector_words_set = set(c.lower() for c in logical_connectors)
    for part in parts:
        if re.match(formula_pattern, part):
            formula = part.strip()
            if formula:
                sentences.append(formula)
        else:
            lines = [l.strip() for l in part.split('\n') if l.strip()]
            buffer = ""
            for line in lines:
                is_heading = re.match(r'^\d+\.\s+\*\*', line)
                is_list_item = re.match(r'^[-*]\s+', line)
                is_structural = is_heading or is_list_item
                is_connector = connector_pattern.match(line)
                buffer_ends_with_colon = buffer.strip().endswith(':')

                if buffer_ends_with_colon:
                    buffer += " " + line
                
                elif is_structural or is_connector:
                    if buffer:
                        sentences.append(buffer)
                    buffer = line
                
                else:
                    buffer += (" " + line).strip()

            if buffer.strip():
                sentences.append(buffer.strip())

    final_sentences = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i].strip()
        if not current_sentence:
            i += 1
            continue
        subs = re.split(r'([.?!])\s+(?=[A-Z])', current_sentence)
        temp_buffer = ""
        sub_sentences = []
        for j, sub in enumerate(subs):
            if j % 2 == 0:
                temp_buffer += sub
            else:
                prev_text = temp_buffer.strip()
                temp_buffer += sub
                is_enumerator = enumerator_pattern.search(prev_text)
                if (not temp_buffer.endswith(':')) and (not is_enumerator):
                    sub_sentences.append(temp_buffer.strip())
                    temp_buffer = ""
        if temp_buffer.strip():
            sub_sentences.append(temp_buffer.strip())

        if not sub_sentences and current_sentence:
            sub_sentences = [current_sentence]

        for s in sub_sentences:
            s_clean = s.strip()
            s_test = s_clean.rstrip('.,').lower()
            
            is_connector_only = s_test in connector_words_set
            is_colon_ended = s_clean.endswith(':')

            if final_sentences:
                last_s_clean = final_sentences[-1].strip()
                last_is_connector = last_s_clean.rstrip('.,').lower() in connector_words_set
                last_is_colon_ended = last_s_clean.endswith(':')

                if last_is_connector or last_is_colon_ended:
                    final_sentences[-1] += " " + s
                    continue

            final_sentences.append(s)

    final_sentences = [re.sub(r'\s+', ' ', s).strip() for s in final_sentences if s.strip()]
    return final_sentences


class ProblemAnswerDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        """
        Args:
            file_path (str): Path to the dataset (JSONL file with {"question": ..., "answer": ...}).
            tokenizer: A tokenizer (e.g., GPT tokenizer) for tokenizing input text.
            max_length (int): Maximum sequence length.
            eos_token_id (int): End-of-sequence token ID.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_token_id = tokenizer.eos_token_id
        json_data = load_jsonl(file_path)
        self.data = []
        for q_a in json_data:
            question = q_a["question"]
            answers = q_a["answer"]
            hints = question
            for answer in answers:
                steps = split_solution_into_sentences(answer)
                for i in range(len(steps)):
                    self.data.append({
                        "hints": hints + " " + " ".join(steps[:i]),
                        "steps": steps[i]
                    })
                hints = hints + " " + " ".join(steps)
                
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        data_entity = self.data[idx]
        hints = data_entity["hints"]
        steps = data_entity["steps"]
        # Tokenize
        hints_tokens = torch.tensor(self.tokenizer.encode(hints, truncation=False)[-256:], dtype=torch.long) # only keep the last 256 tokens of hints
        steps_tokens = torch.tensor(self.tokenizer.encode(steps, max_length=128, truncation=True), dtype=torch.long) # max_length for steps is 128
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