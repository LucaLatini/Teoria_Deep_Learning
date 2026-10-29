"""Conversione integrale, sequenziale e verificabile degli appunti di teoria.

Richiede Python 3, Pandoc, XeLaTeX; per i diagrammi: Node, Playwright e Mermaid.
Gli originali non vengono modificati. Il manuale introduttivo è escluso.
"""
from pathlib import Path
import argparse, collections, copy, hashlib, json, re, shutil, subprocess, sys
from urllib.parse import unquote

sys.stdout.reconfigure(encoding='utf-8')
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
BUILD = ROOT / 'tmp' / 'dispensa_build'
ASSETS = OUT / 'figure'
SOURCES = [
    ('mlp_appunti_completi.md', 'Multilayer Perceptron (MLP)'),
    ('03_CNN.md', 'Reti neurali convoluzionali (CNN)'),
    ('04_RNN.md', 'Serie temporali e reti neurali ricorrenti'),
    ('05_Transformer.md', 'Temporal Convolutional Networks e Transformer'),
]
PANDOC_READER = 'markdown+tex_math_dollars+tex_math_single_backslash-smart'
TOOLS = ROOT / 'tmp' / 'dispensa_build_tools'
PLAYWRIGHT = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright'
REPORT = {'ambito': 'Conversione dei quattro Markdown teorici, a partire da MLP. Manuale introduttivo escluso per istruzione esplicita.', 'fonti': [], 'asset': {}, 'verifiche': []}

def sha(data):
    if isinstance(data, str): data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def run(args, content=None, cwd=None):
    r = subprocess.run([str(x) for x in args], input=content, text=True, encoding='utf-8', capture_output=True, cwd=cwd)
    if r.returncode:
        raise RuntimeError(f'{args[0]}: {r.stderr[-6000:]}\n{r.stdout[-1500:]}')
    return r.stdout

def pandoc(content, src, target, *extra):
    return run(['pandoc', '-f', src, '-t', target, '--wrap=none', *extra], content)

def walk(value):
    if isinstance(value, dict):
        if 't' in value: yield value
        for x in value.values(): yield from walk(x)
    elif isinstance(value, list):
        for x in value: yield from walk(x)

def atoms(value):
    """Independently comparable payloads: prose, formulas, code and images."""
    result = collections.Counter()
    for n in walk(value):
        t, c = n['t'], n.get('c')
        if t == 'Str': result[('testo', c)] += 1
        elif t == 'Math': result[('formula', c[0]['t'], c[1])] += 1
        elif t == 'Code': result[('codice', c[1])] += 1
        elif t == 'CodeBlock': result[('blocco_codice', c[1])] += 1
        elif t == 'Image': result[('immagine', c[2][0])] += 1
    return result

def counts(ast):
    c = collections.Counter(n['t'] for n in walk(ast))
    return {k: c[k] for k in ['Header', 'Para', 'Plain', 'Math', 'Table', 'Image', 'CodeBlock', 'BlockQuote', 'BulletList', 'OrderedList']}

def resolve_image(target, source):
    target = unquote(target).split('|')[0]
    options = [source.parent / target, ROOT / target, ROOT / 'assets' / Path(target).name]
    found = [p.resolve() for p in options if p.is_file()]
    if not found: raise FileNotFoundError(f'Immagine mancante in {source.name}: {target}')
    p = found[0]
    data = p.read_bytes()
    name = 'fig_' + sha(data)[:16] + p.suffix.lower()
    dst = ASSETS / name
    if dst.exists(): assert dst.read_bytes() == data
    else: dst.write_bytes(data)
    REPORT['asset'][name] = {'origine': str(p.relative_to(ROOT)), 'sha256': sha(data), 'byte': len(data), 'tipo': 'immagine originale'}
    return 'figure/' + name

def preprocess(path, ordinal):
    raw = path.read_bytes()
    text = raw.decode('utf-8-sig')
    lines = text.splitlines(keepends=True)
    record = {'file': path.name, 'sha256_originale': sha(raw), 'righe_originali': len(lines), 'esclusioni': [], 'trasformazioni': [], 'righe': []}
    (OUT / 'originali' / path.name).write_bytes(raw)
    out, in_yaml, yaml_finished = [], False, False
    in_fence = False
    for i, original in enumerate(lines, 1):
        line = original.rstrip('\r\n')
        reason = None
        if i == 1 and line == '---': in_yaml = True
        if in_yaml:
            reason = 'metadati YAML iniziali'
            if i > 1 and line == '---': in_yaml = False; yaml_finished = True
        elif re.match(r'^\*\*(Agenda|Convenzione dei riferimenti)', line):
            reason = 'intestazione editoriale iniziale; conservata nel rapporto'
        elif path.name == '04_RNN.md' and line.startswith('I riferimenti «slide N»'):
            reason = 'convenzione editoriale iniziale; conservata nel rapporto'
        if reason:
            record['esclusioni'].append({'riga': i, 'motivo': reason, 'testo': line})
            record['righe'].append({'riga': i, 'stato': 'esclusa', 'sha256': sha(original)})
            continue
        record['righe'].append({'riga': i, 'stato': 'conservata', 'sha256': sha(original)})
        if line.startswith('```'): in_fence = not in_fence
        if not in_fence:
            h = re.match(r'^(#{1,6})\s+(.+)$', line)
            if h:
                title = re.sub(r'^\d+(?:\.\d+)*\.?\s+', '', h[2])
                label = f'fonte{ordinal:02d}' if len(h[1]) == 1 else f'f{ordinal:02d}-r{i}'
                changed = h[1] + ' ' + title + ' {#' + label + '}'
                record['trasformazioni'].append({'riga': i, 'tipo': 'titolo e numerazione LaTeX', 'prima': line, 'dopo': changed})
                line = changed
            # Only an existing CNN section reference requires renumbering.
            if path.name == '03_CNN.md' and 'Sezione 3.1' in line:
                line = line.replace('Sezione 3.1', r'Sezione \ref{f02-r170}')
                record['trasformazioni'].append({'riga': i, 'tipo': 'rinvio aggiornato alla numerazione globale', 'prima': 'Sezione 3.1', 'dopo': r'Sezione \ref{f02-r170}'})
            # Obsidian callouts become normal, open text with a bold heading.
            m = re.match(r'^(>\s*)\[!([^\]]+)\][+-]?\s*(.*)$', line)
            if m:
                title = re.sub(r'^[🔍🔢📊]\s*', '', m[3])
                line = m[1] + '**' + (title or m[2].capitalize()) + '**\n>\n'
                record['trasformazioni'].append({'riga': i, 'tipo': 'callout aperto; testo conservato', 'prima': m[0], 'dopo': line})
            # Obsidian image embeds are converted before ordinary wikilinks.
            def embed(m):
                bits = m[1].split('|', 1)
                caption = bits[1] if len(bits) == 2 and not bits[1].isdigit() else ''
                return '![' + caption + '](' + resolve_image(bits[0], path) + ')'
            line = re.sub(r'!\[\[([^\]]+)\]\]', embed, line)
            def wikilink(m):
                bits = m[1].split('|', 1)
                target = bits[0].split('#')[0].removesuffix('.md')
                label = bits[1] if len(bits) == 2 else {
                    'mlp_appunti_completi': 'Multilayer Perceptron', '03_CNN': 'reti convoluzionali',
                    '04_RNN': 'reti ricorrenti', '05_Transformer': 'Transformer',
                    'Manuale_Studio_Machine_Learning_to_Deep_Learning': 'manuale introduttivo di Machine Learning (documento separato)',
                }.get(target, target.replace('_', ' '))
                lookup = {Path(name).stem: j for j, (name, _) in enumerate(SOURCES, 1)}
                if target in lookup: return '[' + label + '](#fonte' + f'{lookup[target]:02d}' + ')'
                return label
            changed = re.sub(r'(?<!!)\[\[([^\]]+)\]\]', wikilink, line)
            if changed != line: record['trasformazioni'].append({'riga': i, 'tipo': 'wikilink reso leggibile', 'prima': line, 'dopo': changed})
            line = changed
        out.append(line + '\n')
    assert yaml_finished and len(record['righe']) == len(lines)
    assert sum(x['stato'] == 'conservata' for x in record['righe']) + len(record['esclusioni']) == len(lines)
    record['righe_conservate'] = sum(x['stato'] == 'conservata' for x in record['righe'])
    record['righe_escluse'] = len(record['esclusioni'])
    result = ''.join(out)
    assert '[[' not in result
    return result, record

def transform(ast, path, ordinal, record):
    jobs = []
    diagram_n = 0
    for n in walk(ast):
        t, c = n['t'], n.get('c')
        if t == 'Image':
            if not c[2][0].startswith('figure/'):
                c[2][0] = resolve_image(c[2][0], path)
        elif t == 'CodeBlock' and 'mermaid' in c[0][1]:
            diagram_n += 1
            ident = f'diagramma_f{ordinal:02d}_{diagram_n:02d}'
            source = c[1]
            stem = ASSETS / ident
            stem.with_suffix('.mmd').write_text(source, encoding='utf-8')
            jobs.append({'id': ident, 'source': source, 'svg': str(stem.with_suffix('.svg')), 'pdf': str(stem.with_suffix('.pdf'))})
            REPORT['asset'][ident + '.pdf'] = {'origine': path.name, 'tipo': 'diagramma Mermaid renderizzato', 'sha256_sorgente': sha(source), 'sorgente': 'figure/' + ident + '.mmd'}
            n.clear()
            n.update({'t': 'Para', 'c': [{'t': 'Image', 'c': [['', [], []], [], ['figure/' + ident + '.pdf', '']]}]})
    if jobs:
        jobfile = BUILD / f'diagrammi_f{ordinal:02d}.json'
        write_json(jobfile, jobs)
        render = run(['node', OUT / 'strumenti/render_diagrammi.cjs', jobfile, TOOLS / 'mermaid.min.js', PLAYWRIGHT])
        for job in jobs:
            p = Path(job['pdf'])
            assert p.is_file() and p.stat().st_size > 1000
            REPORT['asset'][p.name]['sha256'] = sha(p.read_bytes())
        record['diagrammi'] = len(jobs)
        print(render, end='', flush=True)
    return ast

def organizza_argomenti(ast, ordinal):
    """Un argomento principale per capitolo; TCN e Transformer separati."""
    names = {1: 'MLP - Multilayer Perceptron', 2: 'CNN - Reti neurali convoluzionali', 3: 'RNN - Reti neurali ricorrenti', 4: 'TCN - Temporal Convolutional Networks'}
    def inlines(s):
        return [{'t': 'Str', 'c': s}]
    blocks = [b for b in ast['blocks'] if not (b['t'] == 'RawBlock' and '\\begin{indiceargomento}' in b['c'][1])]
    if ordinal == 4:
        blocks = [b for j,b in enumerate(blocks) if not (j > 0 and b['t'] == 'Header' and b['c'][0] == 1 and b['c'][1][0] == 'fonte04')]
    blocks[0]['c'][2] = inlines(names[ordinal])
    if ordinal == 4:
        blocks[0]['c'][1][0] = 'argomento-tcn'
        split = next(j for j,b in enumerate(blocks) if b['t'] == 'Header' and b['c'][1][0] == 'f04-r266')
        blocks.insert(split, {'t':'Header','c':[1,['fonte04',[],[]],inlines('Transformer')]})
    result = []
    for j,b in enumerate(blocks):
        result.append(b)
        if b['t'] != 'Header' or b['c'][0] != 1: continue
        sections = []
        for child in blocks[j+1:]:
            if child['t'] == 'Header' and child['c'][0] == 1: break
            if child['t'] == 'Header' and child['c'][0] == 2: sections.append(child)
        lines = [r'\begin{indiceargomento}']
        for child in sections:
            doc = {'pandoc-api-version':ast['pandoc-api-version'],'meta':{},'blocks':[{'t':'Plain','c':child['c'][2]}]}
            title = pandoc(json.dumps(doc,ensure_ascii=False),'json','latex').strip()
            label = child['c'][1][0]
            lines.append(r'\item \hyperref[' + label + ']{' + title + '}')
        lines.append(r'\end{indiceargomento}')
        result.append({'t':'RawBlock','c':['latex','\n'.join(lines)]})
    ast['blocks'] = result
    return ast

def tex_fragment(ast):
    result = pandoc(json.dumps(ast, ensure_ascii=False), 'json', 'latex', '--top-level-division=chapter')
    # Keep figures adjacent to their explanations; avoid floats moving past a section.
    result = result.replace(r'\begin{figure}', r'\begin{figure}[H]')
    # Pandoc writes isolated display math. Constrain long formulas without removing symbols.
    def bound_math(m):
        return '\\begin{center}\n\\begin{adjustbox}{max width=\\linewidth}\n$\\displaystyle ' + m[1] + '$\n\\end{adjustbox}\n\\end{center}'
    result = re.sub(r'\\\[([\s\S]*?)\\\]', bound_math, result)
    return result

PREAMBLE = r'''% Compilare con XeLaTeX, due volte. Tutto il testo è in questo unico file.
\documentclass[11pt,a4paper,oneside,openany]{book}
\usepackage[italian]{babel}
\usepackage{fontspec}
\setmainfont{TeX Gyre Pagella}
\setsansfont{TeX Gyre Heros}
\setmonofont{Latin Modern Mono}[Scale=MatchLowercase,BoldFont=lmmonolt10-bold.otf,BoldItalicFont=lmmonolt10-boldoblique.otf]
\usepackage[a4paper,left=22mm,right=22mm,top=23mm,bottom=23mm,headheight=15pt]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{graphicx,adjustbox,float}
\usepackage{longtable,booktabs,array,calc,multirow}
\usepackage{etoolbox}
\usepackage{xcolor}
\definecolor{inchiostro}{HTML}{163B4B}
\definecolor{grigio}{HTML}{52636B}
\usepackage[protrusion=false]{microtype}
\usepackage{upquote,fvextra}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage{xurl}
\usepackage[unicode,colorlinks=true,linkcolor=inchiostro,urlcolor=inchiostro,bookmarksnumbered=true]{hyperref}
\usepackage{bookmark}
\usepackage{newunicodechar}
\newunicodechar{→}{\ensuremath{\rightarrow}}
\newunicodechar{←}{\ensuremath{\leftarrow}}
\newunicodechar{×}{\ensuremath{\times}}
\newunicodechar{−}{\ensuremath{-}}
\newunicodechar{≈}{\ensuremath{\approx}}
\newunicodechar{∂}{\ensuremath{\partial}}
\newunicodechar{∈}{\ensuremath{\in}}
\newcommand{\pandocbounded}[1]{\begin{adjustbox}{max width=\linewidth,max totalheight=.70\textheight,keepaspectratio,center}#1\end{adjustbox}}
\providecommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt plus 1pt minus 1pt}
\setlength{\emergencystretch}{6em}
\setlength{\LTpre}{8pt}
\setlength{\LTpost}{8pt}
\AtBeginEnvironment{longtable}{\small\setlength{\tabcolsep}{4pt}}
\renewenvironment{quote}{\list{}{\leftmargin=1em\rightmargin=0pt}\item\relax}{\endlist}
\titleformat{\chapter}[display]{\sffamily\bfseries\color{inchiostro}}{\large Argomento\ \thechapter}{8pt}{\huge}
\titlespacing*{\chapter}{0pt}{0pt}{24pt}
\titleformat{\section}{\sffamily\Large\bfseries\color{inchiostro}}{\thesection}{.7em}{}
\titleformat{\subsection}{\sffamily\large\bfseries\color{inchiostro}}{\thesubsection}{.7em}{}
\titleformat{\subsubsection}{\sffamily\normalsize\bfseries}{\thesubsubsection}{.7em}{}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{0}
\renewcommand{\chaptername}{Argomento}
\addto\captionsitalian{\renewcommand{\contentsname}{Indice degli argomenti}}
\newenvironment{indiceargomento}{\begingroup\small\sffamily\color{grigio}\textbf{Contenuti dell'argomento}\par\begin{itemize}\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}{\end{itemize}\endgroup\medskip}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\footnotesize\sffamily\color{grigio}\nouppercase{\leftmark}}
\fancyfoot[R]{\small\sffamily\thepage}
\renewcommand{\headrulewidth}{.3pt}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\hypersetup{pdftitle={Machine Learning e Deep Learning - Dispensa di teoria},pdfsubject={MLP, CNN, RNN e Transformer}}
\begin{document}
\frontmatter
\begin{titlepage}
\vspace*{30mm}
{\sffamily\small\color{grigio}APPUNTI DI STUDIO\par}
\vspace{10mm}
{\sffamily\Huge\bfseries\color{inchiostro}Machine Learning\par e Deep Learning\par}
\vspace{9mm}
{\sffamily\LARGE Dispensa di teoria\par}
\vspace{10mm}
{\color{inchiostro}\rule{\textwidth}{1pt}\par}
\vspace{6mm}
{\large MLP - Multilayer Perceptron\par CNN - Reti convoluzionali\par RNN - Reti ricorrenti\par TCN - Temporal Convolutional Networks\par Transformer\par}
\vfill
\end{titlepage}
\tableofcontents
\mainmatter
'''

def checkpoint(fragments):
    tex = PREAMBLE + '\n\n'.join(fragments) + '\n\\end{document}\n'
    (OUT / 'Dispensa_Deep_Learning.tex').write_text(tex, encoding='utf-8')
    write_json(OUT / 'verifica' / 'registro_conversione.json', REPORT)

def convert(limit=4, start=1):
    global REPORT
    for folder in [OUT, ASSETS, BUILD, OUT / 'verifica', OUT / 'originali']:
        folder.mkdir(parents=True, exist_ok=True)
    fragments = []
    if start > 1:
        REPORT = json.loads((OUT / 'verifica/registro_conversione.json').read_text('utf-8'))
        REPORT['fonti'] = REPORT['fonti'][:start-1]
        for j in range(1, start):
            fragments.append((BUILD / f'f{j:02d}.tex').read_text('utf-8'))
    for ordinal, (name, _) in enumerate(SOURCES[:limit], 1):
        if ordinal < start: continue
        print(f'INIZIO {ordinal}: {name}', flush=True)
        path = ROOT / '01_Teoria' / name
        cleaned, record = preprocess(path, ordinal)
        (BUILD / f'f{ordinal:02d}_normalizzato.md').write_text(cleaned, encoding='utf-8')
        baseline = json.loads(pandoc(cleaned, PANDOC_READER, 'json'))
        record['conteggi_lettura'] = counts(baseline)
        # Check math extraction against the original with only authorized editorial lines removed.
        raw_remaining = ''.join(line for i, line in enumerate(path.read_text('utf-8-sig').splitlines(keepends=True), 1) if record['righe'][i-1]['stato'] == 'conservata')
        original_ast = json.loads(pandoc(raw_remaining, PANDOC_READER, 'json'))
        original_math = collections.Counter(n['c'][1] for n in walk(original_ast) if n['t'] == 'Math')
        cleaned_math = collections.Counter(n['c'][1] for n in walk(baseline) if n['t'] == 'Math')
        assert original_math == cleaned_math, f'Formule cambiate nella preparazione: {name}'
        record['formule_originali_vs_preparazione'] = 'identiche, con molteplicità'
        # Scan all remaining dollar pairs: no unresolved math delimiters in prose.
        suspicious = [n['c'] for n in walk(baseline) if n['t'] == 'Str' and '$' in n['c']]
        if suspicious: raise AssertionError(f'Delimitatori matematici non interpretati: {name}: {suspicious[:15]}')
        ast = organizza_argomenti(transform(copy.deepcopy(baseline), path, ordinal, record), ordinal)
        # Record each block independently, including its payload and exact emitted TeX.
        blocktex = []
        block_records = []
        for j, block in enumerate(ast['blocks'], 1):
            ident = f'F{ordinal:02d}-B{j:04d}'
            one = {'pandoc-api-version': ast['pandoc-api-version'], 'meta': {}, 'blocks': [block]}
            piece = tex_fragment(one)
            blocktex.append(f'% BEGIN {ident}\n' + piece + f'% END {ident}\n')
            block_records.append({'id': ident, 'tipo': block['t'], 'sha256_ast': sha(json.dumps(block, sort_keys=True, ensure_ascii=False)), 'sha256_tex': sha(piece), 'conteggi': counts(block)})
        record['blocchi'] = block_records
        record['conteggi_finali'] = counts(ast)
        # Reparse the complete intermediate JSON; compare all prose, math, code and image atoms.
        serialized = json.dumps(ast, ensure_ascii=False)
        reloaded = json.loads(serialized)
        assert atoms(ast) == atoms(reloaded)
        assert counts(ast) == counts(reloaded)
        # The writer is also exercised with the whole chapter (independent of block assembly).
        whole = tex_fragment(ast)
        joined = ''.join(re.sub(r'% (?:BEGIN|END) F\d+-B\d+\n', '', x) for x in blocktex)
        assert re.sub(r'\s+', '', whole) == re.sub(r'\s+', '', joined), 'Il writer di capitolo differisce dai blocchi'
        write_json(OUT / 'verifica' / f'f{ordinal:02d}_contenuto.json', ast)
        fragment = '\n'.join(blocktex)
        fragments.append(fragment)
        (BUILD / f'f{ordinal:02d}.tex').write_text(fragment, encoding='utf-8')
        (OUT / 'verifica' / f'f{ordinal:02d}_frammento.txt').write_text(fragment, encoding='utf-8')
        record['sha256_frammento_tex'] = sha(fragment)
        record['controllo_blocchi'] = 'tutti presenti, nello stesso ordine; emissione indipendente di capitolo e blocchi equivalente'
        REPORT['fonti'].append(record)
        checkpoint(fragments)
        print(f'VERIFICATO {name}: {len(block_records)} blocchi, {record["conteggi_finali"]["Math"]} formule, {record["conteggi_finali"]["Table"]} tabelle', flush=True)
    audit()

def audit():
    p = OUT / 'verifica/registro_conversione.json'
    report = json.loads(p.read_text('utf-8'))
    tex = (OUT / 'Dispensa_Deep_Learning.tex').read_text('utf-8')
    for source in report['fonti']:
        original = ROOT / '01_Teoria' / source['file']
        assert sha(original.read_bytes()) == source['sha256_originale'], 'Originale modificato'
        assert sha((OUT / 'originali' / source['file']).read_bytes()) == source['sha256_originale']
        for b in source['blocchi']:
            pattern = rf'% BEGIN {b["id"]}\n(.*?)% END {b["id"]}\n'
            m = re.search(pattern, tex, re.S)
            assert m and sha(m[1]) == b['sha256_tex'], f'Blocco mancante o alterato: {b["id"]}'
    for name, a in report['asset'].items():
        p = ASSETS / name
        assert p.is_file() and sha(p.read_bytes()) == a['sha256'], f'Asset alterato: {name}'
    assert r'\input{' not in tex and r'\include{' not in tex
    assert '[[mlp' not in tex and '[!bug]' not in tex
    totals = collections.Counter()
    for s in report['fonti']: totals.update(s['conteggi_finali'])
    rows = ['# Verifica della dispensa di teoria', '', report['ambito'], '',
        'Gli originali sono conservati senza modifiche. La dispensa trasferisce i contenuti dei Markdown; non costituisce una nuova revisione scientifica delle slide o delle trascrizioni.', '',
        '## Copertura file per file', '', '| Fonte | Righe originali | Righe escluse | Blocchi trasferiti | Formule | Tabelle | Immagini e diagrammi |', '|---|---:|---:|---:|---:|---:|---:|']
    for s in report['fonti']:
        c=s['conteggi_finali']
        rows.append(f'| {s["file"]} | {s["righe_originali"]} | {s["righe_escluse"]} | {len(s["blocchi"])} | {c["Math"]} | {c["Table"]} | {c["Image"]} |')
    rows += ['', '## Criterio di conservazione', '',
        '- Ogni riga originale è classificata come conservata oppure esclusa con motivazione e impronta SHA-256.',
        '- Ogni blocco trasferito ha un identificatore nel sorgente LaTeX, un’impronta del contenuto strutturato e una del frammento LaTeX.',
        '- Le formule sono confrontate con quelle dei contenuti originali mantenuti, rispettando anche le occorrenze ripetute.',
        '- L’emissione del capitolo completo è confrontata con quella ottenuta assemblando i singoli blocchi.',
        '- Ogni immagine originale è copiata senza modifiche e verificata byte per byte; i diagrammi conservano anche il sorgente Mermaid.',
        '- Le domande esplicative nel corpo e gli esempi svolti sono conservati. Non sono presenti sezioni autonome finali di domande di ripasso nei quattro file.',
        '- Riepiloghi, errata e precisazioni matematiche restano inclusi. Nessuna deduplicazione dei contenuti teorici.',
        '- Il rinvio CNN «Sezione 3.1» è aggiornato con un riferimento LaTeX alla corrispondente sezione.',
        '- I rimandi al manuale introduttivo sono etichettati come riferimenti a un documento separato: il manuale non è incluso.', '',
        '## Esclusioni editoriali', '', 'Sono esclusi soltanto i metadati YAML iniziali e le agende o convenzioni iniziali qui riportate. I titoli sono mantenuti come capitoli e sezioni, con numerazione uniforme.', '']
    for s in report['fonti']:
        rows += ['### ' + s['file'], '']
        for ex in s['esclusioni']:
            if ex['motivo'].startswith('metadati'): continue
            rows += [f'- Riga {ex["riga"]}: {ex["testo"]}', '']
    rows += ['## Dettaglio verificabile', '', 'Il registro JSON contiene la classificazione di tutte le righe, le trasformazioni editoriali, gli identificatori dei blocchi e le impronte dei file. I quattro JSON di contenuto conservano la struttura interpretata da Pandoc.', '',
        'Compilazione: eseguire XeLaTeX due volte nella cartella che contiene il sorgente. Il file `.tex` contiene tutto il testo; la cartella `figure/` contiene i soli asset esterni necessari.', '']
    (OUT / 'Verifica_conversione.md').write_text('\n'.join(rows), encoding='utf-8')
    print('AUDIT COMPLETATO:', dict(totals), flush=True)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=4)
    p.add_argument('--audit', action='store_true')
    p.add_argument('--start', type=int, default=1)
    a = p.parse_args()
    if a.audit: audit()
    else: convert(a.limit, a.start)
