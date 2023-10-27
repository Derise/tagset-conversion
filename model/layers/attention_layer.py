import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=config['word_encoder_output_dim'],
                                               num_heads=config['attention_nhead'],
                                               dropout=config['attention_dropout'], bias=False, batch_first=True)
        self.layer_norm = nn.LayerNorm(normalized_shape=config['word_encoder_output_dim'])

    def forward(self, hidden, target_lel, mask):
        # hidden = [batch * words, 2 * config.word_encoder_hidden_dim]
        # target_lel = [batch * words, max_grams_len, hidden_dim]

        output, output_weights = self.attention(query=hidden.unsqueeze(1), key=target_lel, value=target_lel,
                                                key_padding_mask=mask, average_attn_weights=False)
        # output = [batch * word, 2 * config.word_encoder_hidden_dim]
        output = output.squeeze(1)
        output = self.layer_norm(hidden + output)

        return output
