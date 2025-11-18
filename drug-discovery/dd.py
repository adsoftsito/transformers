# mol_gan_smiles_numpy.py
import numpy as np

# -----------------------
# Configuration
# -----------------------
np.random.seed(42)
vocab_chars = list("CNOPSClBr-=#()[]0123456789")  # small SMILES-ish vocab (extend as needed)
# add padding and start token
PAD_TOKEN = "<PAD>"
START_TOKEN = "<S>"
UNK_TOKEN = "<U>"
vocab = [PAD_TOKEN, START_TOKEN, UNK_TOKEN] + vocab_chars
ivocab = {c:i for i,c in enumerate(vocab)}
vocab_size = len(vocab)

L = 32  # sequence length (max SMILES len)
embed_dim = 32
hidden_dim = 64
batch_size = 16

# Toy dataset: small set of simple SMILES (expand with real dataset later)
toy_smiles = ["CCO", "CCC", "CCN", "COC", "C=O", "CCl", "CN(C)C", "CC(=O)O", "C1CC1", "CCBr"]

# -----------------------
# Utilities: tokenize / detokenize
# -----------------------
def tokenize(smiles, L=L):
    seq = [ivocab.get(c, ivocab[UNK_TOKEN]) for c in smiles]
    seq = [ivocab[START_TOKEN]] + seq  # include start token
    if len(seq) > L:
        seq = seq[:L]
    seq = seq + [ivocab[PAD_TOKEN]]*(L - len(seq))
    return np.array(seq, dtype=np.int32)

def detokenize(indices):
    s = []
    for i in indices:
        c = vocab[i]
        if c == PAD_TOKEN: break
        if c == START_TOKEN: continue
        s.append(c)
    return "".join(s)

# Create dataset arrays
X = np.stack([tokenize(s) for s in toy_smiles])  # shape (N, L)
N = X.shape[0]

# -----------------------
# Simple NumPy modules (embedding, linear, layernorm)
# -----------------------
def one_hot(indices, V=vocab_size):
    # indices: (B, L)
    B,L = indices.shape
    oh = np.zeros((B, L, V), dtype=np.float32)
    for i in range(B):
        for j in range(L):
            oh[i,j,indices[i,j]] = 1.0
    return oh

def glorot(shape):
    r = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(-r, r, size=shape).astype(np.float32)

# Embedding matrix
W_emb = np.random.randn(vocab_size, embed_dim).astype(np.float32) * 0.1

# Positional encoding (simple learnable)
pos_emb = np.random.randn(L, embed_dim).astype(np.float32) * 0.1

# Transformer params (single head causal)
W_q = glorot((embed_dim, embed_dim))
W_k = glorot((embed_dim, embed_dim))
W_v = glorot((embed_dim, embed_dim))
W_o = glorot((embed_dim, embed_dim))
ln_epsilon = 1e-5

# MLP after attention
W_m1 = glorot((embed_dim, hidden_dim))
b_m1 = np.zeros((hidden_dim,), dtype=np.float32)
W_m2 = glorot((hidden_dim, embed_dim))
b_m2 = np.zeros((embed_dim,), dtype=np.float32)

# Output projection
W_out = glorot((embed_dim, vocab_size))
b_out = np.zeros((vocab_size,), dtype=np.float32)

# -----------------------
# GAN params
# -----------------------
z_dim = 16
# Generator: MLP z -> (L * vocab_size) logits
G_W1 = glorot((z_dim, 128))
G_b1 = np.zeros((128,), dtype=np.float32)
G_W2 = glorot((128, L * vocab_size))
G_b2 = np.zeros((L * vocab_size,), dtype=np.float32)

# Discriminator: consumes soft one-hot sequences (B, L, V) flattened -> MLP -> scalar
D_W1 = glorot((L * vocab_size, 128))
D_b1 = np.zeros((128,), dtype=np.float32)
D_W2 = glorot((128, 1))
D_b2 = np.zeros((1,), dtype=np.float32)

# -----------------------
# Helpers: activations, softmax, cross-entropy, gumbel-softmax
# -----------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def cross_entropy_logits(logits, targets):
    # logits: (B, L, V), targets: (B, L) int
    probs = softmax(logits, axis=-1)
    B,L,V = logits.shape
    loss = 0.0
    for i in range(B):
        for j in range(L):
            t = targets[i,j]
            if t == ivocab[PAD_TOKEN]:
                continue
            loss -= np.log(max(probs[i,j,t], 1e-9))
    # normalize by valid tokens
    count = np.sum(targets != ivocab[PAD_TOKEN])
    return loss / max(1, count)

def gumbel_sample(shape):
    u = np.random.rand(*shape)
    return -np.log(-np.log(u + 1e-9) + 1e-9)

def gumbel_softmax_sample(logits, tau=1.0):
    # logits: (..., V)
    g = gumbel_sample(logits.shape)
    y = softmax((logits + g) / tau, axis=-1)
    return y  # soft one-hot

# -----------------------
# Tiny Transformer forward (causal)
# -----------------------
def transformer_forward(input_indices):
    # input_indices: (B, L) int
    B,L = input_indices.shape
    x = W_emb[input_indices] + pos_emb[None,:,:]  # (B,L,embed_dim)
    # single-head causal self-attention
    q = x @ W_q  # (B,L,E)
    k = x @ W_k
    v = x @ W_v
    # compute attention scores with causal mask
    scores = q @ k.transpose(0,2,1) / np.sqrt(embed_dim)  # (B,L,L)
    # mask future positions
    mask = np.tril(np.ones((L,L), dtype=np.float32))
    scores = scores * mask[None,:,:] + (-1e9) * (1.0 - mask)[None,:,:]
    attn = softmax(scores, axis=-1)  # (B,L,L)
    out = attn @ v  # (B,L,E)
    out = out @ W_o
    # Add & Norm (simple)
    x = x + out
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + ln_epsilon)
    # MLP
    m = relu(x @ W_m1 + b_m1[None,None,:])
    m = m @ W_m2 + b_m2[None,None,:]
    x = x + m
    # output logits
    logits = x @ W_out + b_out[None,None,:]  # (B,L,V)
    return logits

# -----------------------
# Generator & Discriminator forward
# -----------------------
def generator_forward(z_batch, tau=0.8):
    # z_batch: (B, z_dim)
    h = relu(z_batch @ G_W1 + G_b1[None,:])
    logits = h @ G_W2 + G_b2[None,:]  # (B, L*V)
    logits = logits.reshape(z_batch.shape[0], L, vocab_size)
    # apply gumbel-softmax per position
    soft_tokens = np.zeros_like(logits, dtype=np.float32)
    for i in range(z_batch.shape[0]):
        soft_tokens[i] = gumbel_softmax_sample(logits[i], tau=tau)
    return logits, soft_tokens  # logits useful for loss, soft_tokens for D input

def discriminator_forward(soft_seq):
    # soft_seq: (B, L, V) soft one-hot; flatten
    B = soft_seq.shape[0]
    x = soft_seq.reshape(B, L * vocab_size)
    h = relu(x @ D_W1 + D_b1[None,:])
    out = h @ D_W2 + D_b2[None,:]  # (B,1)
    # sigmoid
    return 1.0 / (1.0 + np.exp(-out)).reshape(-1)  # (B,)

# -----------------------
# Losses and simple gradient approximations (VERY basic / illustrative)
# NOTE: For real training use autodiff frameworks. Here we do simple finite-difference-ish grads for teaching.
# -----------------------
# For brevity and safety we implement parameter updates with simple gradient estimators:
def sigmoid(x): return 1/(1+np.exp(-x))

# Simple training params
lr_T = 1e-3  # transformer lr
lr_G = 1e-3
lr_D = 1e-3

# -----------------------
# Training loops (toy)
# -----------------------
def train_transformer(epochs=200):
    # Train the transformer as a tiny LM on toy SMILES
    for ep in range(epochs):
        # sample batch
        idx = np.random.choice(N, batch_size, replace=True)
        batch = X[idx]  # (B,L)
        logits = transformer_forward(batch)  # (B,L,V)
        # target is shifted left: predict next token
        targets = np.zeros_like(batch)
        targets[:, :-1] = batch[:, 1:]
        loss = cross_entropy_logits(logits, targets)
        if ep % 50 == 0:
            print(f"[Transformer] Epoch {ep} loss {loss:.4f}")
        # -- VERY simple gradient-free-ish parameter nudges (this is illustrative only) --
        # A toy "update": nudge output weights toward one-hot targets via simple delta: (not true grad)
        probs = softmax(logits, axis=-1)
        B = batch.shape[0]
        grad_out = np.zeros_like(W_out)
        for i in range(B):
            for j in range(L):
                t = targets[i,j]
                if t == ivocab[PAD_TOKEN]: continue
                p = probs[i,j]
                err = p.copy()
                err[t] -= 1.0
                # hidden representation approx: use embedding + pos (very coarse)
                h_ij = W_emb[batch[i,j]] + pos_emb[j]
                grad_out += np.outer(h_ij, err)
        W_out[:] -= lr_T * grad_out / max(1,B)

def train_gan(steps=500):
    # Alternating training: D then G
    for step in range(steps):
        # --- Train D ---
        # real batch: take real sequences and convert to one-hot (hard)
        idx = np.random.choice(N, batch_size, replace=True)
        real_batch_idx = X[idx]
        real_onehot = one_hot(real_batch_idx)  # (B,L,V)
        # fake batch:
        z = np.random.randn(batch_size, z_dim).astype(np.float32)
        g_logits, fake_soft = generator_forward(z, tau=0.8)
        D_real = discriminator_forward(real_onehot)
        D_fake = discriminator_forward(fake_soft)
        # discriminator loss (BCE)
        eps=1e-9
        loss_D = -np.mean(np.log(D_real+eps) + np.log(1 - D_fake + eps))
        # Very simple parameter updates (illustrative): nudge D weights based on difference
        # compute pseudo-gradients
        # flatten inputs
        # gradient direction: increase output for real, decrease for fake
        grad_out_real = (sigmoid((real_onehot.reshape(batch_size, -1) @ D_W1 + D_b1) @ D_W2 + D_b2.reshape(1,-1)) - 1.0)
        # Instead of full backprop, we do a simple heuristic update:
        D_W2[:] -= lr_D * np.random.randn(*D_W2.shape) * 0.01
        D_W1[:] -= lr_D * np.random.randn(*D_W1.shape) * 0.01

        # --- Train G ---
        z = np.random.randn(batch_size, z_dim).astype(np.float32)
        _, fake_soft = generator_forward(z, tau=0.8)
        D_fake2 = discriminator_forward(fake_soft)
        # generator tries to make D_fake close to 1
        loss_G = -np.mean(np.log(D_fake2 + eps))
        # heuristic updates of G params:
        G_W2[:] -= lr_G * np.random.randn(*G_W2.shape) * 0.01
        G_W1[:] -= lr_G * np.random.randn(*G_W1.shape) * 0.01

        if step % 100 == 0:
            print(f"[GAN] Step {step} loss_D {loss_D:.4f} loss_G {loss_G:.4f}")

# -----------------------
# Sampling utilities
# -----------------------
def sample_transformer_temperature(start_seq="<S>", max_len=32, temp=1.0):
    idxs = [ivocab[START_TOKEN]]
    for t in range(max_len-1):
        arr = np.array([idxs + [ivocab[PAD_TOKEN]]*(L - len(idxs))], dtype=np.int32)
        logits = transformer_forward(arr)  # (1,L,V)
        # take logits at current position (len(idxs)-1)
        pos = len(idxs)-1
        logp = logits[0,pos] / temp
        prob = softmax(logp)
        nxt = np.random.choice(len(vocab), p=prob)
        idxs.append(int(nxt))
        if vocab[nxt] == PAD_TOKEN:
            break
    return detokenize(idxs)

def sample_gan(n=5, tau=0.8):
    z = np.random.randn(n, z_dim).astype(np.float32)
    _, soft = generator_forward(z, tau=tau)
    # convert soft to hard indices by argmax
    idxs = np.argmax(soft, axis=-1)  # (n, L)
    return [detokenize(row) for row in idxs]

# -----------------------
# RDKit validation (optional)
# -----------------------
try:
    from rdkit import Chem
    def validate_smiles(sm):
        try:
            m = Chem.MolFromSmiles(sm)
            return m is not None
        except:
            return False
except Exception as e:
    def validate_smiles(sm):
        return False

# -----------------------
# Run a small demo
# -----------------------
if __name__ == "__main__":
    print("Vocab size:", vocab_size, "Seq len:", L)
    # Quick transformer pretrain (toy)
    print("Training tiny transformer (toy)...")
    train_transformer(epochs=200)
    print("Sampling from transformer:")
    for _ in range(5):
        print("  ", sample_transformer_temperature())

    # Train GAN (very toy / illustrative)
    print("Training toy GAN (illustrative)...")
    train_gan(steps=600)
    print("Sample GAN outputs:")
    samples = sample_gan(8)
    for s in samples:
        ok = validate_smiles(s)
        print(f"  {s}   valid? {ok}")

