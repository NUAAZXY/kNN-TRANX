import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Effective Approaches to Attention-based Neural Machine Translation by Luong et al.
    Concat method
    """
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, mask):
        # hidden = [1, batch size, dec hid dim]
        # encoder_outputs = [nl nl_length, batch size, enc hid dim * 2]

        nl_length = encoder_outputs.shape[0]

        # repeat decoder hidden state nl_len times (T encoder hidden states, 1 decoder hidden state)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.repeat(1, nl_length, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        # hidden = [batch size, nl len, dec hid dim]
        # encoder_outputs = [batch size, nl len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # energy = [batch size, src len, dec hid dim]
        # weighted sum of the energy across all encoder hidden states

        attention = self.v(energy).squeeze(2)

        # attention= [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim=1)
