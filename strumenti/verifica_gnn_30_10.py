"""Verifica indipendente dei piccoli esempi ricostruiti per la lezione 30/10."""
from pathlib import Path
import json
import numpy as np

out = Path(__file__).resolve().parents[1]
tests = {}
z = np.array([2., 0., -1.])
p = np.exp(z) / np.exp(z).sum()
np.testing.assert_allclose(p, [.843795, .114195, .042010], atol=5e-7)
tests['classificatore_nodi'] = {'logit': z.tolist(), 'probabilita': p.tolist(),
                              'cross_entropy': float(-np.log(p[0]))}
tests['loss_transduttiva'] = float(-(np.log(.8) + np.log(.5)) / 2)
q = 1 / (1 + np.exp(-4.))
tests['link'] = {'logit': 4., 'probabilita': q,
                 'bce_presente': float(-np.log(q)), 'bce_assente': float(-np.log(1-q))}
H = np.array([[1., 0.], [0., 2.], [3., 1.]])
s = H @ np.array([np.log(2), 0.])
alpha = np.exp(s) / np.exp(s).sum()
g = alpha @ H
np.testing.assert_allclose(g, [26/11, 10/11])
tests['pooling'] = {'media': H.mean(0).tolist(), 'score': s.tolist(),
                     'pesi': alpha.tolist(), 'pesata': g.tolist(),
                     'gerarchico': [H[:2].mean(0).tolist(), H[2].tolist()]}
H = np.array([[1., 0.], [0., 2.], [-1., 1.]])
a = np.array([1., -1., 1., 1.])
raw = np.array([[np.r_[x, y] @ a for y in H] for x in H])
e = np.where(raw >= 0, raw, .2*raw)
mask = np.array([[1,1,0], [1,1,1], [0,1,1]], dtype=bool)
e = np.where(mask, e, -np.inf)
ex = np.exp(e - np.max(e, axis=1, keepdims=True))
P = ex / ex.sum(1, keepdims=True)
pre, post = P @ H, np.maximum(P @ H, 0)
np.testing.assert_allclose(P, [[.268941,.731059,0], [.328933,.401760,.269307],
                              [0,.598688,.401312]], atol=5e-7)
np.testing.assert_allclose(pre, [[.268941,1.462117], [.059625,1.072827],
                                [-.401312,1.598688]], atol=5e-7)
np.testing.assert_allclose(P.sum(1), 1)
assert np.all(P[~mask] == 0)
tests['GAT'] = {'score_prima_attivazione': raw.tolist(),
                'score_mascherati': [[float(v) if np.isfinite(v) else '-inf' for v in row] for row in e],
                'coefficienti': P.tolist(), 'prima_relu': pre.tolist(),
                'dopo_relu': post.tolist(), 'somma_righe': P.sum(1).tolist()}
tests['oversmoothing'] = (np.ones((2,2))/2 @ np.array([[0.], [2.]])).tolist()
tests['esito'] = 'PASS'
(out/'verifica/verifica_numerica_GNN_30-10.json').write_text(
    json.dumps(tests, ensure_ascii=False, indent=2), encoding='utf-8')
print('PASS: classificazione nodi, loss transduttiva, link, pooling, GAT e smoothing.')
