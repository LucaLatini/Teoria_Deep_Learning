"""Assembla il preambolo aggiornato, compila e registra gli avvisi LaTeX."""
import argparse, importlib.util, json, re, shutil, subprocess
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--chapters', type=int, default=4)
p.add_argument('--passes', type=int, default=2)
a = p.parse_args()
spec = importlib.util.spec_from_file_location('converti', Path(__file__).with_name('converti.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.REPORT = json.loads((m.OUT / 'verifica/registro_conversione.json').read_text('utf-8'))
parts = [(m.BUILD / f'f{i:02d}.tex').read_text('utf-8') for i in range(1, a.chapters + 1)]
partial = a.chapters < len(m.REPORT['fonti'])
stem = 'verifica_parziale' if partial else 'Dispensa_Deep_Learning'
if partial:
    texpath = m.BUILD / (stem + '.tex')
    texpath.write_text(m.PREAMBLE + '\n\n'.join(parts) + '\n\\end{document}\n', encoding='utf-8')
else:
    m.checkpoint(parts)
    m.audit()
    texpath = m.OUT / (stem + '.tex')
for i in range(a.passes):
    r = subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=' + m.BUILD.as_posix(), texpath.as_posix()], cwd=m.OUT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    (m.BUILD / f'xelatex_pass{i+1}.txt').write_text(r.stdout + r.stderr, encoding='utf-8')
    print(f'Compilazione {i+1}: {r.returncode}', flush=True)
    if r.returncode:
        print(r.stdout[-4500:])
        raise SystemExit(r.returncode)
log = (m.BUILD / (stem + '.log')).read_text('utf-8', errors='replace')
warnings = [x for x in log.splitlines() if any(t in x for t in ['Overfull', 'Missing character', 'undefined', 'Error', 'Warning'])]
(m.OUT / 'verifica/compilazione.json').write_text(json.dumps({'passaggi': a.passes, 'avvisi': warnings}, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n'.join(warnings), flush=True)
if not partial:
    shutil.copy2(m.BUILD / (stem + '.pdf'), m.OUT / 'Dispensa_Deep_Learning.pdf')
