import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken

# Hiperparámetros exactos que usaste para entrenar
block_size = 512
vocab_size = 50257
n_embd = 768
n_head = 12
n_layer = 12
dropout = 0.0 # Apagado para inferencia

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        total_embd = num_heads * head_size
        self.c_attn = nn.Linear(total_embd, 3 * total_embd, bias=False)
        self.c_proj = nn.Linear(total_embd, total_embd)

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MarioLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, idx, max_new_tokens, temperature=0.7, top_p=0.8, top_k=40, repetition_penalty=1.3):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            with torch.no_grad():
                logits = self(idx_cond)

            logits = logits[:, -1, :] / temperature

            # Penalización de repetición
            for b in range(logits.size(0)):
                seen_tokens = idx[b].tolist()
                for token_id in set(seen_tokens):
                    if logits[b, token_id] > 0:
                        logits[b, token_id] /= repetition_penalty
                    else:
                        logits[b, token_id] *= repetition_penalty

            # Top-K
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-P
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            for b in range(logits.size(0)):
                indices_to_remove = sorted_indices[b][sorted_indices_to_remove[b]]
                logits[b, indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    def decodificar_respuesta(ids_generados, encoder_name="gpt2"):

        enc = tiktoken.get_encoding(encoder_name)

        # 🔥 FIX: asegurar lista
        if isinstance(ids_generados, torch.Tensor):
            tokens = ids_generados.view(-1).tolist()
        else:
            tokens = ids_generados

        texto_traducido = enc.decode(tokens)

        return texto_traducido

    # --- Ejemplo de integración en tu bloque de Streamlit ---
    def procesar_salida_mario(generado_idx, tokens_entrada, enc_local):

        # 🔥 FIX: asegurar lista
        if isinstance(generado_idx, torch.Tensor):
            generado_idx = generado_idx.view(-1).tolist()

        if isinstance(tokens_entrada, torch.Tensor):
            tokens_entrada = tokens_entrada.tolist()

        respuesta_total = enc_local.decode(generado_idx)

        longitud_prompt = len(enc_local.decode(tokens_entrada))
        full_response = respuesta_total[longitud_prompt:].strip()

        full_response = full_response.split("Usuario:")[0].split("Asistente:")[0].strip()

        return full_response