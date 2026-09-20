"""Analytic/numeric check (no model): softmax attention is count-weighted across distinct key
directions; (a) ridge leverage normalisation only removes multiplicity for exact duplicates or N>d_k,
(b) kernel soft-multiplicity normalisation removes it also for near-duplicates when N<d_k.
Reference = same query with one copy of each direction."""
import numpy as np
np.random.seed(1)
d = 64
def softmax(x): e = np.exp(x - x.max()); return e / e.sum()
def soft_mult(K, tau):
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True); C = Kn @ Kn.T
    return 1 + (np.exp(tau * (C - 1)).sum(1) - 1)
def leverage(K, lam_rel):
    G = K.T @ K; lam = lam_rel * np.trace(G) / d
    return np.einsum('nd,dn->n', K, np.linalg.solve(G + lam * np.eye(d), K.T))
A = np.random.randn(d); A /= np.linalg.norm(A)
B = np.random.randn(d); B -= (B @ A) * A; B /= np.linalg.norm(B)
q = (A + B) * np.sqrt(d)
gen = [np.random.randn(d) for _ in range(10)]
K = np.stack([k / np.linalg.norm(k) for k in [A, B] + gen]); w = softmax(K @ q / np.sqrt(d))
print(f'REFERENCE (1 A + 1 B + 10 generic): mass A={w[0]:.3f} B={w[1]:.3f} generic={w[2:].sum():.3f}')
for c in [1.0, 0.98, 0.95, 0.90, 0.80]:
    Ks = []
    for _ in range(5):
        n = np.random.randn(d); n -= (n @ A) * A; n /= np.linalg.norm(n); Ks.append(c * A + np.sqrt(1 - c ** 2) * n)
    K = np.stack([k / np.linalg.norm(k) for k in Ks + [B] + gen]); s = K @ q / np.sqrt(d)
    w0 = softmax(s)
    print(f'\n5 near-copies of A (cos={c}) + 1 B + 10 generic | softmax: A-group={w0[:5].sum():.3f} B={w0[5]:.3f} generic={w0[6:].sum():.3f}')
    for tau in [4, 8, 16]:
        m = soft_mult(K, tau); w = softmax(s - np.log(m))
        print(f'   kernel-multiplicity tau={tau:<3} (rho=1): m~A={m[:5].mean():4.2f} m~B={m[5]:4.2f} | A-group={w[:5].sum():.3f} B={w[5]:.3f} generic={w[6:].sum():.3f}')
    for lam in [0.01, 0.1]:
        l = leverage(K, lam); w = softmax(s + np.log(l))
        print(f'   ridge-leverage lam_rel={lam:<4} (rho=1): lev A~={l[:5].mean():4.2f} lev B={l[5]:4.2f} | A-group={w[:5].sum():.3f} B={w[5]:.3f} generic={w[6:].sum():.3f}')

# --- Anisotropy check: a strong common direction shared by all keys makes uncentered cosine
# multiplicity nearly constant across tokens (bias cancels in softmax); per-sentence centering restores it.
np.random.seed(3)
u = np.random.randn(d); u /= np.linalg.norm(u)
A = np.random.randn(d); A -= (A @ u) * u; A /= np.linalg.norm(A)
B = np.random.randn(d); B -= (B @ u) * u; B -= (B @ A) * A; B /= np.linalg.norm(B)
def keys_aniso(cA=0.9):
    Ks = []
    for _ in range(5):
        n = np.random.randn(d); n -= (n @ u) * u; n -= (n @ A) * A; n /= np.linalg.norm(n)
        Ks.append(3 * u + cA * A + np.sqrt(1 - cA ** 2) * n)
    Ks.append(3 * u + B)
    for _ in range(10):
        n = np.random.randn(d); n -= (n @ u) * u; n /= np.linalg.norm(n); Ks.append(3 * u + n)
    return np.stack(Ks)
def mult(K, tau, center):
    if center: K = K - K.mean(0, keepdims=True)
    Kn = K / np.linalg.norm(K, axis=1, keepdims=True); C = Kn @ Kn.T
    return 1 + (np.exp(tau * (C - 1)).sum(1) - 1)
K = keys_aniso()
print('\nANISOTROPY (common component 3u on every key, content unit-norm, 5 near-copies of A at cos 0.9):')
for center in [False, True]:
    m = mult(K, 8, center)
    print(f'   centering={center!s:5}: m~A={m[:5].mean():.2f} m~B={m[5]:.2f} m~generic={m[6:].mean():.2f}')
