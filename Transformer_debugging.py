import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Config
D_MODEL = 128
N_HEADS = 4
D_FF = 512
ENC_LAYERS = 2
DEC_LAYERS = 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttentionSimple(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)

    def forward(self, q_in, k_in, v_in, attn_mask=None):
        batch_size, q_len, _ = q_in.size()
        batch_size, k_len, _ = k_in.size()

        Q = self.q_lin(q_in)
        K = self.k_lin(k_in)
        V = self.v_lin(v_in)

        Qh = Q.view(batch_size, q_len, self.nhead, self.d_head).permute(0, 2, 1, 3)
        Kh = K.view(batch_size, k_len, self.nhead, self.d_head).permute(0, 2, 1, 3)
        Vh = V.view(batch_size, k_len, self.nhead, self.d_head).permute(0, 2, 1, 3)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, float('-1e9'))
        attn_weights = F.softmax(scores, dim=-1)

        attn_output = torch.matmul(attn_weights, Vh)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, q_len, self.d_model)

        out = self.out_lin(attn_output)
        return {
            'Q': Q, 'K': K, 'V': V,
            'Qh': Qh, 'Kh': Kh, 'Vh': Vh,
            'scores_before_softmax': scores,
            'attn_weights': attn_weights,
            'attn_output_heads': attn_output,
            'attn_output': out
        }


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.mha = MultiHeadAttentionSimple(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        enc_in = x
        attn_res = self.mha(x, x, x)
        res1 = x + attn_res['attn_output']
        ln1 = self.ln1(res1)
        ff_in = ln1
        ff1 = self.ff1(ff_in)
        ff1_act = F.relu(ff1)
        ff2 = self.ff2(ff1_act)
        res2 = ln1 + ff2
        ln2 = self.ln2(res2)
        return ln2, {
            'enc_attn': attn_res, 'ln1': ln1,
            'ff1': ff1, 'ff2': ff2, 'ln2': ln2
        }


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super().__init__()
        self.masked_mha = MultiHeadAttentionSimple(d_model, nhead)
        self.cross_mha = MultiHeadAttentionSimple(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.ln3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, tgt_mask=None):
        dec_in = x

        masked_attn_res = self.masked_mha(x, x, x, attn_mask=tgt_mask)
        res1 = x + masked_attn_res['attn_output']
        ln1 = self.ln1(res1)

        cross_attn_res = self.cross_mha(ln1, enc_out, enc_out)
        res2 = ln1 + cross_attn_res['attn_output']
        ln2 = self.ln2(res2)

        ff_in = ln2
        ff1 = self.ff1(ff_in)
        ff1_act = F.relu(ff1)
        ff2 = self.ff2(ff1_act)
        res3 = ln2 + ff2
        ln3 = self.ln3(res3)

        return ln3, {
            'masked_attn': masked_attn_res, 'ln1': ln1,
            'cross_attn': cross_attn_res, 'ln2': ln2,
            'ff1': ff1, 'ff2': ff2, 'ln3': ln3
        }


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)])

    def forward(self, x):
        out = x
        traces = []
        for layer in self.layers:
            out, trace = layer(out)
            traces.append(trace)
        return out, traces


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)])

    def forward(self, x, enc_out, tgt_mask=None):
        out = x
        traces = []
        for layer in self.layers:
            out, trace = layer(out, enc_out, tgt_mask)
            traces.append(trace)
        return out, traces


class SimpleStackedTransformer(nn.Module):
    def __init__(self, vocab_size=10000, d_model=D_MODEL, nhead=N_HEADS, d_ff=D_FF,
                 enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.encoder = Encoder(enc_layers, d_model, nhead, d_ff)
        self.decoder = Decoder(dec_layers, d_model, nhead, d_ff)
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, src_tokens, tgt_tokens, tgt_mask=None):
        raw_src = src_tokens
        raw_tgt = tgt_tokens

        emb_weight = self.embed.weight
        src_emb = self.embed(src_tokens)
        tgt_emb = self.embed(tgt_tokens)
        src_emb_pos = self.pos(src_emb)

        enc_out, enc_traces = self.encoder(src_emb_pos)
        tgt_emb_pos = self.pos(tgt_emb)
        dec_out, dec_traces = self.decoder(tgt_emb_pos, enc_out, tgt_mask)

        final_out = dec_out
        logits = self.proj(final_out)
        _ = logits[0, 0, :10]

        return logits, {'enc_traces': enc_traces, 'dec_traces': dec_traces}


if __name__ == "__main__":
    torch.manual_seed(0)
    model = SimpleStackedTransformer(vocab_size=10000)

    src = torch.tensor([[511, 723, 845, 932, 678]])   # ("the poet writes beautiful verses")
    tgt = torch.tensor([[812, 459, 511, 390, 275]])   # ("words touch the human soul")

    tgt_len = tgt.size(1)
    attn_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool)).unsqueeze(0).unsqueeze(0)

    logits, trace = model(src, tgt, tgt_mask=attn_mask)

    a = 1