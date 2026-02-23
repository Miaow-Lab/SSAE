import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM
from sentenceSAE import Autoencoder

class MyModel(nn.Module):
    def __init__(self, tokenizer, sparsity_factor, activation = nn.ReLU(), init_from=('Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-0.5B'), phase=1):
        super().__init__()
        self.tokenizer = tokenizer
        self.sparsity_factor = sparsity_factor
        self.encoder = AutoModel.from_pretrained(init_from[0])
        self.hints_encoder = AutoModel.from_pretrained(init_from[0])
        self.decoder = AutoModelForCausalLM.from_pretrained(init_from[1])

        new_vocab_size = len(tokenizer)
        print(f"Resizing embeddings to {new_vocab_size}...")
        self.encoder.resize_token_embeddings(new_vocab_size)
        self.hints_encoder.resize_token_embeddings(new_vocab_size)
        self.decoder.resize_token_embeddings(new_vocab_size)
        self.n_inputs = self.encoder.config.hidden_size
        self.n_latents = self.n_inputs * self.sparsity_factor
        print(f'n_inputs: {self.n_inputs}, n_latents: {self.n_latents}')
        self.autoencoder = Autoencoder(n_latents=self.n_latents, n_inputs=self.n_inputs, sparsity_factor=self.sparsity_factor, activation=activation)
        self.mean_mlp = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_latents//8),
            nn.LeakyReLU(),
            nn.Linear(self.n_latents//8, self.n_latents//2),
            nn.LeakyReLU(),
            nn.Linear(self.n_latents//2, self.n_latents),  # mean
        )
        self.var_mlp = nn.Sequential(
            nn.Linear(self.n_inputs, self.n_inputs),
            nn.LeakyReLU(),
            nn.Linear(self.n_inputs, self.n_latents),
            nn.Linear(self.n_latents, self.n_latents),  # logvar
        )
        self.projection_mlp = nn.Sequential(
            nn.Linear(self.n_latents, self.n_latents),
            nn.ReLU(),
            nn.Linear(self.n_latents, self.n_latents),
        )
        self.phase = phase
        if phase == 1:
            self.var_mlp.requires_grad_(False)
            self.mean_mlp.requires_grad_(False)
            self.hints_encoder.requires_grad_(False)
        elif phase == 2:
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.autoencoder.requires_grad_(False)
            self.projection_mlp.requires_grad_(False)
        elif phase == 3:
            self.hints_encoder.requires_grad_(False)
            self.mean_mlp.requires_grad_(False)
            self.encoder.requires_grad_(False)
            self.decoder.requires_grad_(False)
            self.autoencoder.requires_grad_(False)
            self.projection_mlp.requires_grad_(False)
        
    def forward(self, input_ids, attention_mask, hints_sep_ids, hints_sep_attention_masks):
        """
        input_ids: (batch_size, seq_len)
        attention_mask: (batch_size, seq_len), where 1 = real token, 0 = padding
        hints_sep_ids: (batch_size, hints_sep_pad_len)
        hints_sep_attention_masks: (batch_size, hints_sep_pad_len), where 1 = real token, 0 = padding
        return:
            latents: (batch_size, 1, n_latents)
            loss_sparsity_nll: float
            logits: (batch_size, seq_len+sparsity_factor-1, vocab_size)
        """
        batch_size = input_ids.shape[0]
        if self.phase == 1:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state # shape: (batch_size, seq_len, n_inputs)
            last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
            latents = self.autoencoder(last_token_embeddings)
            latents = F.normalize(latents, p=2, dim=-1)  # L1 normalize latents
            loss_sparsity = latents.abs().sum(dim=(1, 2)) # shape: (batch_size)
            noise = torch.randn_like(latents) * 0.01
            latents = latents + noise
            recons = self.projection_mlp(latents) # latents shape: [batch, 1, n_latents]
            recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs) # (batch, sparsity_factor, n_inputs)
            logits = self.decode(recons, input_ids, attention_mask)
            return latents, loss_sparsity.sum(), logits #!
        elif self.phase == 2:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state # shape: (batch_size, seq_len, n_inputs)
            last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
            latents = self.autoencoder(last_token_embeddings) # latents shape: (batch_size, 1, n_latents)
            latents = F.normalize(latents, p=2, dim=-1)  # L1 normalize latents
            
            hints_encoder_outputs = self.hints_encoder(hints_sep_ids, attention_mask=hints_sep_attention_masks).last_hidden_state # shape: (batch_size, hints_sep_pad_len, n_inputs)
            hints_last_token_embeddings = self.get_last_token_embeddings(hints_encoder_outputs, hints_sep_attention_masks) # shape: (batch_size, 1, n_inputs)
            mean = self.mean_mlp(hints_last_token_embeddings) # shape: (batch_size, 1, n_latents)
            log_var = self.var_mlp(hints_last_token_embeddings) # shape: (batch_size, 1, n_latents)
            var = torch.exp(log_var)
            log_likelihood = - ((latents - mean) ** 2 / (2 * var)) - 0.5 * torch.log(2 * torch.pi * var)
            nll_loss = -log_likelihood.mean()
            mean_loss = (latents - mean) ** 2
            mean_error = torch.sqrt(mean_loss.mean())
            return nll_loss, mean_error

        elif self.phase == 3:
            encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state # shape: (batch_size, seq_len, n_inputs)
            last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
            latents = self.autoencoder(last_token_embeddings) # latents shape: (batch_size, 1, n_latents)
            hints_encoder_outputs = self.hints_encoder(hints_sep_ids, attention_mask=hints_sep_attention_masks).last_hidden_state # shape: (batch_size, hints_sep_pad_len, n_inputs)
            hints_avg_token_embeddings = self.get_avg_token_embeddings(hints_encoder_outputs, hints_sep_attention_masks) # shape: (batch_size, 1, n_inputs)
            mean = self.mean_mlp(hints_avg_token_embeddings) # shape: (batch_size, 1, n_latents)
            log_var = self.var_mlp(hints_avg_token_embeddings) # shape: (batch_size, 1, n_latents)
            var = torch.exp(log_var)
            log_likelihood = - ((latents - mean) ** 2 / (2 * var)) - 0.5 * torch.log(2 * torch.pi * var)
            nll_loss = -log_likelihood.mean()
            mean_loss = (latents - mean) ** 2
            mean_error = torch.sqrt(mean_loss.mean())
            return nll_loss, mean_error
            
            
    @torch.no_grad()
    def sample_Tr(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
        latents = self.autoencoder(last_token_embeddings)
        latents = F.normalize(latents, p=2, dim=-1)  # L1 normalize latents
        return latents
    
    @torch.no_grad()
    def sample_hint_emb(self, hints_sep_ids, hints_sep_attention_masks):
        hints_encoder_outputs = self.hints_encoder(hints_sep_ids, attention_mask=hints_sep_attention_masks).last_hidden_state
        hints_last_token_embeddings = self.get_last_token_embeddings(hints_encoder_outputs, hints_sep_attention_masks)
        return hints_last_token_embeddings

    def decode(self, recons, input_ids, attention_mask):
        """
        recons: (batch_size, sparsity_factor, n_inputs)
        inputs_ids: (batch_size, seq_len)
        sep_pos: (batch_size)
        attention_mask: (batch_size, seq_len), where 1 = real token, 0 = padding
        return: 
            all_logits:(batch_size, seq_len+sparsity_factor-1, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        # teacher forcing
        last_token_position = (attention_mask.sum(dim=1)-1).long() # shape: (batch_size)
        # remove last_token_position
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        mask[torch.arange(batch_size), last_token_position] = False
        without_last_token_ids = input_ids[mask].reshape(batch_size, seq_len-1) # hints+sep+shift
        # without_last_token_ids = input_ids[:, :-1] # hints+sep+shift
        embedding_layer = self.decoder.get_input_embeddings()
        without_last_token_embeds = embedding_layer(without_last_token_ids)
        inputs_embs = torch.cat([recons, without_last_token_embeds], dim=1) # shape: (batch_size, seq_len+sparsity_factor-1, n_inputs)
        recons_attention_mask = torch.ones((batch_size, self.sparsity_factor-1), dtype=torch.long, device=device)
        attention_mask = torch.cat([recons_attention_mask, attention_mask], dim=1) # shape: (batch_size, seq_len+sparsity_factor-1)
        decoder_outputs = self.decoder(inputs_embeds=inputs_embs, attention_mask=attention_mask)
        return decoder_outputs.logits
    
    
    @torch.no_grad()
    def manual_generate_sentence(self, input_ids, attention_mask, hints_sep_ids, hints_sep_attention_masks, temperature, top_k, top_p, max_new_tokens=256):
        """
        input hints+sep+step, get ground latents, decode with ground latents
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len), where 1 = real token, 0 = padding
            hints_sep_ids: (batch_size, hints_sep_pad_len)
            hints_sep_attention_masks: (batch_size, hints_sep_pad_len), where 1 = real token, 0 = padding

        Returns:
            latents: (batch_size, 1, n_latents)
            decode_text_ids: (batch_size, max_new_tokens)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask).last_hidden_state
        last_token_embeddings = self.get_last_token_embeddings(encoder_outputs, attention_mask)
        latents = self.autoencoder(last_token_embeddings)
        latents = F.normalize(latents, p=2, dim=-1)  # L1 normalize latents
        recons = self.projection_mlp(latents) # recons shape: [batch, 1, n_latents]
        # reshape recons to (batch, sparsity_factor, n_inputs)
        recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)
        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device)
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        topk_records = []
        
        embedding_layer = self.decoder.get_input_embeddings()
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
        for _ in range(max_new_tokens):
            hints_sep_embeds = embedding_layer(hints_sep_ids)
            inputs_embs = torch.cat([recons, hints_sep_embeds], dim=1)
            
            recons_attention_mask = torch.ones((batch_size, self.sparsity_factor), dtype=torch.long, device=device)
            decode_attention_mask = torch.cat([recons_attention_mask, hints_sep_attention_masks], dim=1)
            
            decoder_outputs = self.decoder(inputs_embeds=inputs_embs, attention_mask=decode_attention_mask)
            
            valid_lengths = (hints_sep_attention_masks == 1).sum(dim=1) - 1
            decode_last_valid_token_indices = valid_lengths + self.sparsity_factor
            last_token_logits = decoder_outputs.logits[torch.arange(batch_size), decode_last_valid_token_indices, :]
            
            scaled_logits = last_token_logits.float()
            if temperature > 0:
                scaled_logits = scaled_logits / temperature

            probs = F.softmax(scaled_logits, dim=-1)  # (batch_size, vocab_size)

            top10_probs, top10_token_ids = torch.topk(
                probs, k=10, dim=-1
            )  # (batch_size, 10)

            topk_records.append({
                "token_ids": top10_token_ids,   # (batch_size, 10)
                "probs": top10_probs             # (batch_size, 10)
            })

            next_tokens = torch.stack([decode_logits(last_token_logits[i], temperature, top_k, top_p) for i in range(batch_size)], dim=0).unsqueeze(1)
            newly_finished = (next_tokens.squeeze(1) == self.tokenizer.eos_token_id) & (unfinished_sequences == 1)
            unfinished_sequences[newly_finished] = 0

            tokens_to_add = next_tokens.clone().squeeze(1)
            tokens_to_add[unfinished_sequences == 0] = pad_token_id
            decode_text_ids = torch.cat([decode_text_ids, tokens_to_add.unsqueeze(1)], dim=1)
            
            if unfinished_sequences.max() == 0:
                break
            orig_seq_len = hints_sep_ids.shape[1]
            new_seq_len = orig_seq_len + 1
            new_hints_sep_ids = torch.full((batch_size, new_seq_len), pad_token_id, dtype=hints_sep_ids.dtype, device=device)
            new_hints_sep_attention_masks = torch.zeros((batch_size, new_seq_len), dtype=hints_sep_attention_masks.dtype, device=device)
            indices_src = torch.arange(orig_seq_len, device=device).unsqueeze(0)
            indices_dest = torch.arange(new_seq_len, device=device).unsqueeze(0)
            insert_positions = valid_lengths + 1
            mask_before_dest = indices_dest < insert_positions.unsqueeze(1)
            mask_before_src = indices_src < insert_positions.unsqueeze(1)
            new_hints_sep_ids[mask_before_dest] = hints_sep_ids[mask_before_src]
            new_hints_sep_attention_masks[mask_before_dest] = hints_sep_attention_masks[mask_before_src]
            unfinished_mask = (unfinished_sequences == 1)
            active_indices = torch.where(unfinished_mask)[0]
            if active_indices.numel() > 0:
                new_hints_sep_ids[active_indices, insert_positions[unfinished_mask]] = next_tokens[unfinished_mask].squeeze(-1)
                new_hints_sep_attention_masks[active_indices, insert_positions[unfinished_mask]] = 1
            mask_after_src = (indices_src >= insert_positions.unsqueeze(1))
            mask_after_dest = (indices_dest >= (insert_positions.unsqueeze(1) + 1))
            new_hints_sep_ids[mask_after_dest] = hints_sep_ids[mask_after_src]
            new_hints_sep_attention_masks[mask_after_dest] = hints_sep_attention_masks[mask_after_src]
            hints_sep_ids = new_hints_sep_ids
            hints_sep_attention_masks = new_hints_sep_attention_masks
        return latents, decode_text_ids, topk_records
    
    
    @torch.no_grad()
    def generate_sentence(self, hints_sep_ids, hints_sep_attention_masks, temperature, top_k, top_p, max_new_tokens=256):
        """
        input hints+sep, sample latents, decode with sampling latents
        Args:
            hints_sep_ids: (batch_size, hints_sep_pad_len)
            hints_sep_attention_masks: (batch_size, hints_sep_pad_len), where 1 = real token, 0 = padding
        """
        batch_size = hints_sep_ids.shape[0]
        device = hints_sep_ids.device
        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id
        hints_encoder_outputs = self.hints_encoder(hints_sep_ids, attention_mask=hints_sep_attention_masks).last_hidden_state
        hints_last_token_embeddings = self.get_last_token_embeddings(hints_encoder_outputs, hints_sep_attention_masks)
        mean = self.mean_mlp(hints_last_token_embeddings)
        log_var = self.var_mlp(hints_last_token_embeddings)
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(mean, device=device)
        sample = mean + epsilon * std
        sample = torch.clamp(sample, min=0)
        recons = self.projection_mlp(sample)
        recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)

        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device)
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        embedding_layer = self.decoder.get_input_embeddings()
        for _ in range(max_new_tokens):
            hints_sep_embeds = embedding_layer(hints_sep_ids)
            inputs_embs = torch.cat([recons, hints_sep_embeds], dim=1)
            
            recons_attention_mask = torch.ones((batch_size, self.sparsity_factor), dtype=torch.long, device=device)
            decode_attention_mask = torch.cat([recons_attention_mask, hints_sep_attention_masks], dim=1)
            
            decoder_outputs = self.decoder(inputs_embeds=inputs_embs, attention_mask=decode_attention_mask)
            
            valid_lengths = (hints_sep_attention_masks == 1).sum(dim=1) - 1
            decode_last_valid_token_indices = valid_lengths + self.sparsity_factor
            last_token_logits = decoder_outputs.logits[torch.arange(batch_size), decode_last_valid_token_indices, :]
            
            next_tokens = torch.stack([decode_logits(last_token_logits[i], temperature, top_k, top_p) for i in range(batch_size)], dim=0).unsqueeze(1)
            newly_finished = (next_tokens.squeeze(1) == self.tokenizer.eos_token_id) & (unfinished_sequences == 1)
            unfinished_sequences[newly_finished] = 0

            tokens_to_add = next_tokens.clone().squeeze(1)
            tokens_to_add[unfinished_sequences == 0] = pad_token_id
            decode_text_ids = torch.cat([decode_text_ids, tokens_to_add.unsqueeze(1)], dim=1)
            
            if unfinished_sequences.max() == 0:
                break
            orig_seq_len = hints_sep_ids.shape[1]
            new_seq_len = orig_seq_len + 1
            new_hints_sep_ids = torch.full((batch_size, new_seq_len), pad_token_id, dtype=hints_sep_ids.dtype, device=device)
            new_hints_sep_attention_masks = torch.zeros((batch_size, new_seq_len), dtype=hints_sep_attention_masks.dtype, device=device)
            indices_src = torch.arange(orig_seq_len, device=device).unsqueeze(0)
            indices_dest = torch.arange(new_seq_len, device=device).unsqueeze(0)
            insert_positions = valid_lengths + 1
            mask_before_dest = indices_dest < insert_positions.unsqueeze(1)
            mask_before_src = indices_src < insert_positions.unsqueeze(1)
            new_hints_sep_ids[mask_before_dest] = hints_sep_ids[mask_before_src]
            new_hints_sep_attention_masks[mask_before_dest] = hints_sep_attention_masks[mask_before_src]
            unfinished_mask = (unfinished_sequences == 1)
            active_indices = torch.where(unfinished_mask)[0]
            if active_indices.numel() > 0:
                new_hints_sep_ids[active_indices, insert_positions[unfinished_mask]] = next_tokens[unfinished_mask].squeeze(-1)
                new_hints_sep_attention_masks[active_indices, insert_positions[unfinished_mask]] = 1
            mask_after_src = (indices_src >= insert_positions.unsqueeze(1))
            mask_after_dest = (indices_dest >= (insert_positions.unsqueeze(1) + 1))
            new_hints_sep_ids[mask_after_dest] = hints_sep_ids[mask_after_src]
            new_hints_sep_attention_masks[mask_after_dest] = hints_sep_attention_masks[mask_after_src]
            hints_sep_ids = new_hints_sep_ids
            hints_sep_attention_masks = new_hints_sep_attention_masks

        return mean, std, sample, decode_text_ids
    
    
    @torch.no_grad()
    def generate_sentence_with_latents(self, hints_sep_ids, hints_sep_attention_masks, latents, temperature, top_k, top_p, max_new_tokens=256):
        """
        Args:
            hints_sep_ids: (batch_size, hints_sep_pad_len)
            hints_sep_attention_masks: (batch_size, hints_sep_pad_len), where 1 = real token, 0 = padding
            latents: (batch_size, 1, n_latents)
        """
        batch_size = latents.shape[0]
        device = latents.device
        
        recons = self.projection_mlp(latents)
        recons = recons.view(batch_size, self.sparsity_factor, self.n_inputs)

        newline_token_id = self.tokenizer.encode("\n")[0]
        decode_text_ids = torch.full((batch_size, 1), newline_token_id, dtype=torch.long, device=device)
        
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)

        if self.tokenizer.pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        else:
            pad_token_id = self.tokenizer.pad_token_id

        for _ in range(max_new_tokens):
            if 'qwen' in self.decoder.config.model_type:
                hints_sep_embeds = self.decoder.model.embed_tokens(hints_sep_ids)
            else:
                hints_sep_embeds = self.decoder.transformer.wte(hints_sep_ids)
            inputs_embs = torch.cat([recons, hints_sep_embeds], dim=1)
            
            recons_attention_mask = torch.ones((batch_size, self.sparsity_factor), dtype=torch.long, device=device)
            decode_attention_mask = torch.cat([recons_attention_mask, hints_sep_attention_masks], dim=1)
            
            decoder_outputs = self.decoder(inputs_embeds=inputs_embs, attention_mask=decode_attention_mask)
            
            valid_lengths = (hints_sep_attention_masks == 1).sum(dim=1) - 1
            decode_last_valid_token_indices = valid_lengths + self.sparsity_factor
            last_token_logits = decoder_outputs.logits[torch.arange(batch_size), decode_last_valid_token_indices, :]
            
            next_tokens = torch.stack([decode_logits(last_token_logits[i], temperature, top_k, top_p) for i in range(batch_size)], dim=0).unsqueeze(1)
            newly_finished = (next_tokens.squeeze(1) == self.tokenizer.eos_token_id) & (unfinished_sequences == 1)
            unfinished_sequences[newly_finished] = 0

            tokens_to_add = next_tokens.clone().squeeze(1)
            tokens_to_add[unfinished_sequences == 0] = pad_token_id
            decode_text_ids = torch.cat([decode_text_ids, tokens_to_add.unsqueeze(1)], dim=1)
            
            if unfinished_sequences.max() == 0:
                break
            orig_seq_len = hints_sep_ids.shape[1]
            new_seq_len = orig_seq_len + 1
            new_hints_sep_ids = torch.full((batch_size, new_seq_len), pad_token_id, dtype=hints_sep_ids.dtype, device=device)
            new_hints_sep_attention_masks = torch.zeros((batch_size, new_seq_len), dtype=hints_sep_attention_masks.dtype, device=device)
            indices_src = torch.arange(orig_seq_len, device=device).unsqueeze(0)
            indices_dest = torch.arange(new_seq_len, device=device).unsqueeze(0)
            insert_positions = valid_lengths + 1
            mask_before_dest = indices_dest < insert_positions.unsqueeze(1)
            mask_before_src = indices_src < insert_positions.unsqueeze(1)
            new_hints_sep_ids[mask_before_dest] = hints_sep_ids[mask_before_src]
            new_hints_sep_attention_masks[mask_before_dest] = hints_sep_attention_masks[mask_before_src]
            unfinished_mask = (unfinished_sequences == 1)
            active_indices = torch.where(unfinished_mask)[0]
            if active_indices.numel() > 0:
                new_hints_sep_ids[active_indices, insert_positions[unfinished_mask]] = next_tokens[unfinished_mask].squeeze(-1)
                new_hints_sep_attention_masks[active_indices, insert_positions[unfinished_mask]] = 1
            mask_after_src = (indices_src >= insert_positions.unsqueeze(1))
            mask_after_dest = (indices_dest >= (insert_positions.unsqueeze(1) + 1))
            new_hints_sep_ids[mask_after_dest] = hints_sep_ids[mask_after_src]
            new_hints_sep_attention_masks[mask_after_dest] = hints_sep_attention_masks[mask_after_src]
            hints_sep_ids = new_hints_sep_ids
            hints_sep_attention_masks = new_hints_sep_attention_masks

        return decode_text_ids
                
    
    def get_last_token_embeddings(self, encoder_outputs, attention_mask):
        batch_size, seq_len, hidden_dim = encoder_outputs.shape
        last_token_indices = (attention_mask.sum(dim=1) - 1).long()
        last_token_embeddings = torch.stack([
            encoder_outputs[i, last_token_indices[i], :] for i in range(batch_size)
        ]).unsqueeze(1) # shape: (batch_size, 1, hidden_dim)
        return last_token_embeddings


    def get_avg_token_embeddings(self, encoder_outputs, attention_mask):
        """
        attention_mask shape: (batch_size, seq_len)
        encoder_outputs shape: (batch_size, seq_len, hidden_dim)
        return: (batch_size, 1, hidden_dim)
        """
        expanded_mask = attention_mask.unsqueeze(-1).expand(encoder_outputs.size()).float()
        masked_embeddings = encoder_outputs * expanded_mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.clamp(expanded_mask.sum(dim=1), min=1e-9)
        avg_embeddings = sum_embeddings / sum_mask
        return avg_embeddings.unsqueeze(1)


def decode_logits(logits, temperature=0.6, top_k=0, top_p=0.95):
    """
    Args:
        logits (torch.Tensor)
        temperature (float)
        top_k (int)
        top_p (float)

    Returns:
        int: sample token ID。
    """
    logits = logits.float()
    # Greedy Decoding
    if temperature == 0:
        return torch.argmax(logits, dim=-1)
    # Apply Temperature
    logits = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(logits, k=top_k)
        min_value_to_keep = values[-1]
        logits[logits < min_value_to_keep] = -float('Inf')
    elif top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = -float('Inf')
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # Sample from the distribution
    next_token_id = torch.multinomial(probs, num_samples=1)
    return next_token_id.squeeze(-1)