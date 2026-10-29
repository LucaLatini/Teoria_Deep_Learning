"""Controllo indipendente: rilegge il LaTeX emesso con il reader LaTeX di Pandoc.

Confronta formule esatte (salvo il solo comando di stile display inserito) e
parole del testo. I numeri non matematici sono riportati separatamente, perché
il reader LaTeX può reinterpretare etichette e numerazioni degli elenchi.
"""
import collections, importlib.util, json, re, unicodedata
from pathlib import Path

spec = importlib.util.spec_from_file_location('converti', Path(__file__).with_name('converti.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def text(value):
    if isinstance(value, dict):
        t, c = value.get('t'), value.get('c')
        if t == 'Str': return c
        if t == 'Code': return c[1]
        if t in ['Math', 'Image', 'RawBlock', 'RawInline', 'Space', 'SoftBreak', 'LineBreak']: return ' '
        return ' '.join(text(v) for v in value.values())
    if isinstance(value, list): return ' '.join(text(v) for v in value)
    return ''

def tokens(ast):
    s = text(ast).replace("'", ' ').replace('’', ' ').replace('‘', ' ')
    return collections.Counter(re.findall(r'\w+', unicodedata.normalize('NFC', s)))

def math(ast):
    return collections.Counter(re.sub(r'^\\displaystyle\s*', '', n['c'][1].strip()) for n in m.walk(ast) if n['t'] == 'Math')

results = []
for i, (name, _) in enumerate(m.SOURCES, 1):
    source = json.loads((m.OUT / f'verifica/f{i:02d}_contenuto.json').read_text('utf-8'))
    fragment = m.BUILD / f'f{i:02d}.tex'
    if not fragment.is_file(): fragment = m.OUT / f'verifica/f{i:02d}_frammento.txt'
    latex = fragment.read_text('utf-8')
    latex = re.sub(r'\\begin\{indiceargomento\}[\s\S]*?\\end\{indiceargomento\}', '', latex)
    # Remove presentation wrappers only, so that unknown-environment arguments
    # do not appear as prose in the independent reader.
    latex = latex.replace(r'\begin{adjustbox}{max width=\linewidth}', '').replace(r'\end{adjustbox}', '')
    latex = latex.replace(r'\textquotesingle', "'")
    latex = re.sub(r'\\pandocbounded\{(\\includegraphics(?:\[[^\]]*\])?\{[^}]*\})\}', r'\1', latex)
    back = json.loads(m.pandoc(latex, 'latex', 'json'))
    before_m, after_m = math(source), math(back)
    assert before_m == after_m, f'Formule diverse nel ritorno LaTeX: {name}: {list((before_m-after_m).items())[:4]}'
    before_w, after_w = tokens(source), tokens(back)
    missing = {w: n for w, n in (before_w-after_w).items() if any(c.isalpha() for c in w)}
    added = {w: n for w, n in (after_w-before_w).items() if any(c.isalpha() for c in w)}
    assert not missing and not added, f'Testo diverso nel ritorno LaTeX: {name}: mancanti {missing}, aggiunti {added}'
    c1, c2 = m.counts(source), m.counts(back)
    assert c1['Image'] == c2['Image'], f'Immagini mancanti nel ritorno LaTeX: {name}'
    assert c1['Table'] == c2['Table'], f'Tabelle mancanti nel ritorno LaTeX: {name}'
    result = {'file': name, 'formule_verificate': sum(before_m.values()), 'parole_verificate': sum(n for w,n in before_w.items() if any(c.isalpha() for c in w)), 'immagini_verificate': c1['Image'], 'tabelle_verificate': c1['Table'], 'differenze_parole': {}, 'numeri_testuali_interpretati_diversamente_dal_reader': dict(before_w-after_w), 'esito': 'PASS'}
    results.append(result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
m.write_json(m.OUT / 'verifica/verifica_indipendente_latex.json', results)
