"""Applica e verifica la sola revisione editoriale concordata il 30/10/2026.

Legge la fotografia precedente in verifica/prima_revisione_finali. Non
riscrive gli originali, non rigenera figure, non modifica altri blocchi TeX.
"""
import collections
import copy
import importlib.util
import json
import re
from pathlib import Path

from finali_editoriali import revisiona, testo

spec = importlib.util.spec_from_file_location('converti', Path(__file__).with_name('converti.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
base = m.OUT / 'verifica/prima_revisione_finali'
m.REPORT = json.loads((base / 'registro_conversione.json').read_text('utf-8'))
current_report = json.loads((m.OUT / 'verifica/registro_conversione.json').read_text('utf-8'))
if [s['sha256_originale'] for s in current_report['fonti']] != [s['sha256_originale'] for s in m.REPORT['fonti']]:
    raise SystemExit('Questo script ricostruisce la revisione storica dei finali. Sono presenti lezioni successive: usare il sorgente e i frammenti attuali, senza ripristinare la fotografia precedente.')

# Riscontro editoriale dei contenuti già presenti, per riga/item dei finali.
# Gli indici sono quelli degli AST precedenti (base zero), non numeri di slide.
# La corrispondenza è semantica: formule equivalenti possono avere notazione
# compatta nel riepilogo e una derivazione estesa nel testo.
summary = {
    1: [(6,31),(34,89),(99,143),(144,148),(151,173),(178,187),
        (205,262),(269,329),(331,346),(349,385)],
    2: [(23,28),(112,118),(160,164),(191,196),(186,189),(200,207),
        (244,249),(253,258),(260,264),(271,276),(268,273),(268,273),
        (308,315),(319,320),(150,151),(352,355),(369,378),(380,384),
        (396,403),(397,420),(426,433)],
    3: [(7,19),(75,87),(54,61),(96,104),(97,105),(98,104),(113,119),
        (181,203),(186,201),(143,155),(143,149),(157,168),(224,243),
        (249,254),(256,271),(262,264)],
    4: [(19,22),(35,37),(41,54),(43,54),(56,62),(68,78),(114,126),
        (136,145),(147,163),(199,215),(222,228),(231,236),(238,245),
        (248,253),(255,276),(129,145),(281,290),(292,297)],
    5: [(55,63),(55,63),(70,74),(71,77),(75,80),(82,88),(82,88),
        (127,132),(141,143),(144,148),(151,155),(176,182),(185,193),
        (205,211),(213,219)],
}
errata = {
    1: [(63,64),(88,89),(88,89),(133,134),(168,169)],
    2: [(91,92),(110,118),(144,151),(248,264),(267,295),(308,320),
        (345,350),(369,384),(396,420),(426,439)],
    3: [(13,15),(23,34),(45,46),(42,44),(48,52),(55,67),(75,87),
        (96,128),(134,140),(143,168),(181,203),(205,213),(215,246),
        (222,226),(256,271),(278,285)],
    4: [(35,39),(14,37),(41,54),(56,62),(68,78),(80,87),(89,95),
        (104,112),(114,126),(129,145),(147,163),(174,176),(179,191),
        (199,228),(217,245),(255,276),(278,297),(308,322),(324,326)],
    5: [(9,14),(16,26),(29,34),(36,40),(57,63),(75,80),(82,88),
        (91,98),(118,123),(141,149),(134,155),(170,182),(185,202),
        (205,215),(213,219)],
}


def rows(block):
    assert block['t'] == 'Table'
    return [row for body in block['c'][4] for row in body[3]]


def coverage(units, ranges, blocks, file_n, category):
    assert len(units) == len(ranges), (file_n, category, len(units), len(ranges))
    records = []
    for k, (unit, (lo, hi)) in enumerate(zip(units, ranges), 1):
        targets = [f'F{file_n:02d}-B{i+1:04d}' for i in range(lo, hi+1)]
        records.append({'voce': k, 'testo_precedente': testo(unit),
                        'riscontro': 'contenuto già presente nel corpo; riepilogo ridondante',
                        'blocchi_corpo': targets,
                        'sha256_riscontro': m.sha(json.dumps(blocks[lo:hi+1], ensure_ascii=False, sort_keys=True))})
    return records


details, fragments, moved_metadata = [], [], []
for ordinal, source in enumerate(m.REPORT['fonti'], 1):
    before = json.loads((base / f'f{ordinal:02d}_contenuto.json').read_text('utf-8'))
    previous_tex = (base / f'f{ordinal:02d}_frammento.txt').read_text('utf-8')
    pieces = dict(re.findall(r'% BEGIN (F\d+-B\d+)\n(.*?)% END \1\n', previous_tex, re.S))
    assert len(pieces) == len(before['blocks'])
    after, indices, changes = revisiona(before)
    stop = len(indices)
    assert m.counts(after)['Image'] == m.counts(before)['Image']
    changed = {c['indice_originale']: c for c in changes}
    kepttex, records = [], []
    for i, block in zip(indices, after['blocks']):
        ident = f'F{ordinal:02d}-B{i+1:04d}'
        if i in changed:
            one = {'pandoc-api-version': after['pandoc-api-version'], 'meta': {}, 'blocks': [block]}
            piece = m.tex_fragment(one)
        else:
            assert block == before['blocks'][i]
            piece = pieces[ident]
        rendered = f'% BEGIN {ident}\n{piece}% END {ident}\n'
        if ordinal == 3 and i == 49:
            # Solo impaginazione: il titolo della nota alla slide 12 non deve
            # restare isolato a fondo pagina. Nessun blocco di testo cambia.
            assert testo(block).startswith('Slide 12 — Schema statistico generale')
            rendered = '\\begin{samepage}\n' + rendered
        if ordinal == 3 and i == 50:
            rendered += '\\end{samepage}\n'
        kepttex.append(rendered)
        records.append({'id': ident, 'tipo': block['t'],
                        'sha256_ast': m.sha(json.dumps(block, sort_keys=True, ensure_ascii=False)),
                        'sha256_tex': m.sha(piece), 'conteggi': m.counts(block)})
    # Ogni formula, figura, tabella, codice ed esempio rimasto nel corpo è
    # identico, con la stessa molteplicità. Cambiano solo etichette e navigazione.
    body_before = [b for i,b in enumerate(before['blocks'][:stop]) if i not in changed]
    body_after = [b for i,b in enumerate(after['blocks']) if i not in changed]
    assert body_before == body_after
    math_before = collections.Counter(n['c'][1] for n in m.walk(before['blocks'][:stop]) if n['t']=='Math')
    math_after = collections.Counter(n['c'][1] for n in m.walk(after['blocks']) if n['t']=='Math')
    assert math_before == math_after
    final_blocks = before['blocks'][stop:]
    tables = [b for b in final_blocks if b['t']=='Table']
    if ordinal == 1:
        summary_units = next(b['c'][1] for b in final_blocks if b['t']=='OrderedList')
        errata_units = next(b['c'][0]['c'] for b in final_blocks
                            if b['t']=='BlockQuote' and b['c'][0]['t']=='BulletList')
    else:
        assert len(tables) == 2
        summary_units, errata_units = map(rows, tables)
    c_summary = coverage(summary_units, summary[ordinal], before['blocks'], ordinal, 'riepilogo')
    c_errata = coverage(errata_units, errata[ordinal], before['blocks'], ordinal, 'precisazioni')
    if ordinal in (3,4):
        metadata = before['blocks'][-1]
        assert metadata['t']=='Para' and testo(metadata).startswith('Le forme trascritte')
        moved_metadata.append({'fonte': source['file'], 'blocco': f'F{ordinal:02d}-B{len(before["blocks"]):04d}',
                               'testo': testo(metadata), 'destinazione': 'rapporto di revisione; termini corretti già usati nel corpo'})
    source['conteggi_prima_revisione_finali'] = copy.deepcopy(source['conteggi_finali'])
    source['blocchi'] = records
    source['conteggi_finali'] = m.counts(after)
    source['controllo_blocchi'] = 'corpo identico byte per byte salvo etichette delle note e indice; finali esclusi su richiesta del 30/10/2026'
    source['revisione_editoriale'] = {
        'data': '2026-10-30', 'blocchi_esclusi': [f'F{ordinal:02d}-B{i+1:04d}' for i in range(stop,len(before['blocks']))],
        'modifiche_etichette_indice': changes,
        'copertura': 'verifica/revisione_finali_2026-10-30.json'}
    fragment = '\n'.join(kepttex)
    source['sha256_frammento_tex'] = m.sha(fragment)
    m.write_json(m.OUT / f'verifica/f{ordinal:02d}_contenuto.json', after)
    (m.BUILD / f'f{ordinal:02d}.tex').write_text(fragment, encoding='utf-8')
    (m.OUT / f'verifica/f{ordinal:02d}_frammento.txt').write_text(fragment, encoding='utf-8')
    fragments.append(fragment)
    details.append({'fonte':source['file'], 'blocchi_corpo_inalterati':len(body_before),
                    'modifiche_ammesse':changes, 'formule_corpo_inalterate':sum(math_after.values()),
                    'riepilogo':c_summary,'precisazioni_finali':c_errata,
                    'conteggi_prima':m.counts(before),'conteggi_dopo':m.counts(after),'esito':'PASS'})
    print(source['file'], len(body_before), 'blocchi del corpo identici;', len(c_summary), 'voci riepilogo;',len(c_errata),'precisazioni ricondotte al corpo', flush=True)
revision = {'data':'2026-10-30','incarico':'Rimuovere i finali autonomi di riepilogo ed errata; conservare quanto espresso nelle fonti e le note locali; nessun altro cambiamento teorico.',
            'solo_impaginazione':[{'blocchi':['F03-B0050','F03-B0051'],'vincolo':'ambiente samepage','scopo':'tenere il titolo della nota alla slide 12 con il suo testo'}],
            'tipo_controllo_copertura':'riscontro editoriale semantico per voce, distinto dalle verifiche automatiche di identità dei blocchi e formule',
            'file':details,'metadati_trasferiti_al_rapporto':moved_metadata}
m.write_json(m.OUT/'verifica/revisione_finali_2026-10-30.json',revision)
m.REPORT['revisioni_editoriali'] = [{'data':'2026-10-30','ambito':revision['incarico'],'rapporto':'verifica/Revisione_finali_2026-10-30.md','registro': 'verifica/revisione_finali_2026-10-30.json'}]
m.REPORT['struttura_argomenti']['verifica'] = 'sei capitoli conservati; indici locali aggiornati per esclusione dei soli finali autorizzati'
m.checkpoint(fragments)
m.audit()
report = ['# Revisione editoriale del 30 ottobre 2026','',revision['incarico'],'',
          '## Perimetro e conservazione','',
          'I cinque Markdown di teoria, le slide, le trascrizioni, le immagini e lo standard Markdown originale non sono modificati. Il controllo riguarda questa revisione editoriale: non viene dichiarata una nuova lettura scientifica di tutte le fonti del corso. La lezione GNN conserva il limite alla slide stampata 28, pagina PDF 29.','',
          'Sono esclusi soltanto i finali autonomi di riepilogo/errata e i loro separatori. Le note locali mantengono il testo preesistente, comprese le espressioni delle fonti quando riportate, gli esempi e le correzioni. La sola etichetta «Errata slide» diventa «Nota sulla/sulle slide». Definizioni, formule, esempi svolti, derivazioni, tabelle didattiche nel corpo, immagini e rinvii restano invariati.','',
          'Le tabelle finali ripetevano contenuti già spiegati: il registro JSON associa ciascuna riga e ciascun punto di riepilogo ai blocchi del corpo. Non è stato necessario spostare tabelle finali intere. Questo riscontro semantico è distinto dal confronto automatico, esatto, dei blocchi conservati.','',
          '| Fonte | Blocchi del corpo identici | Voci di riepilogo riscontrate | Precisazioni riscontrate | Formule del corpo identiche |',
          '|---|---:|---:|---:|---:|']
for d in details:
    report.append(f'| {d["fonte"]} | {d["blocchi_corpo_inalterati"]} | {len(d["riepilogo"])} | {len(d["precisazioni_finali"])} | {d["formule_corpo_inalterate"]} |')
report += ['','## Note sulla trascrizione conservate fuori dal testo di studio','']
for item in moved_metadata:
    report += ['### '+item['fonte'],'',item['testo'],'']
report += ['## Verifiche ripetibili','',
           'Un solo vincolo di impaginazione, l’ambiente `samepage` attorno a `F03-B0050` e `F03-B0051`, tiene il titolo della nota alla slide 12 insieme al suo testo, senza modificare i blocchi né aggiungere contenuto.','',
           'La fotografia precedente è conservata in `verifica/prima_revisione_finali/`. `strumenti/revisiona_finali.py` ripete esclusivamente la revisione autorizzata, conserva gli identificatori e i frammenti LaTeX dei blocchi non modificati e ricostruisce il registro. `strumenti/finali_editoriali.py` applica lo stesso criterio anche nelle future conversioni. Il controllo indipendente `verifica_semantica.py` confronta il nuovo contenuto strutturato con il LaTeX. Compilazione e resa PDF sono verificate separatamente.','']
(m.OUT/'verifica/Revisione_finali_2026-10-30.md').write_text('\n'.join(report),encoding='utf-8')
