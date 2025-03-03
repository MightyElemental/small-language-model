import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dropout=0.1, max_len=5000):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            num_layers (int): Number of decoder layers.
            dropout (float): Dropout probability.
            max_len (int): Maximum sequence length.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        # Dummy memory for the transformer decoder.
        # TransformerDecoder expects an encoder output. Here we use a learned parameter
        # that is broadcasted to all batches.
        self.dummy_memory = nn.Parameter(torch.zeros(1, 1, d_model))
        
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with -inf."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, tgt, tgt_mask=None):
        """
        Forward function used during training.
        
        Args:
            tgt (Tensor): Target token indices with shape (batch_size, seq_len).
            tgt_mask (Tensor): Optional causal mask of shape (seq_len, seq_len).
            
        Returns:
            logits (Tensor): Output logits with shape (batch_size, seq_len, vocab_size).
        """
        # Embed tokens and add positional encodings.
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        # Transformer modules expect shape (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(x.size(0)).to(x.device)
        
        # Create a dummy memory of shape (memory_seq_len, batch_size, d_model).
        # Here we use a single “dummy” token for the encoder output.
        batch_size = x.size(1)
        memory = self.dummy_memory.expand(1, batch_size, self.d_model)
        
        x = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        x = x.transpose(0, 1)  # (batch_size, seq_len, d_model)
        logits = self.decoder(x)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=20, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively.
        
        Args:
            input_ids (Tensor): Starting sequence of token ids with shape (batch_size, seq_len).
            max_new_tokens (int): Maximum number of tokens to generate.
            temperature (float): Temperature for sampling.
            top_k (int, optional): If provided, limits sampling to the top_k tokens.
            
        Returns:
            Tensor: The extended sequence including generated tokens.
        """
        generated = input_ids
        for _ in range(max_new_tokens):
            seq_len = generated.size(1)
            tgt_mask = self.generate_square_subsequent_mask(seq_len).to(generated.device)
            logits = self.forward(generated, tgt_mask=tgt_mask)
            # Focus on the last token's logits.
            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None:
                # Top-k filtering: keep only the top_k logits
                values, indices = torch.topk(next_token_logits, top_k)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(1, indices, values)
                next_token_logits = filtered_logits

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
        return generated
