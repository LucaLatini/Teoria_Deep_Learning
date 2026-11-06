"""Intervento autorizzato il 30/10/2026: finali ridondanti e note locali.

Non modifica il corpo delle spiegazioni né le formule. I Markdown restano
originali; la revisione appartiene alla sola destinazione LaTeX.
"""
import copy
import re


def testo(value):
    if isinstance(value, dict):
        kind = value.get('t')
        if kind == 'Str':
            return value['c']
        if kind in ('Space', 'SoftBreak', 'LineBreak'):
            return ' '
        if kind in ('Math', 'Code'):
            return value['c'][1]
        if kind == 'RawBlock':
            return ''
        return testo(value.get('c', []))
    if isinstance(value, list):
        return ''.join(testo(x) for x in value)
    return ''


def revisiona(ast):
    """Restituisce AST, indici originali mantenuti e sole modifiche ammesse.

    La prima sezione finale Riepilogo e le successive Errata vengono escluse.
    Non tocca richiami, ricapitolazioni intermedie o termini come hidden state.
    Un nuovo capitolo dopo un riepilogo non può essere escluso per errore.
    """
    result = copy.deepcopy(ast)
    blocks = result['blocks']
    stops = [i for i, b in enumerate(blocks)
             if b['t'] == 'Header' and b['c'][0] == 2
             and re.match(r'^Riepilogo(?:\s|$)', testo(b), re.I)]
    if not stops:
        return result, list(range(len(blocks))), []
    assert len(stops) == 1, 'Verificare manualmente più sezioni di riepilogo.'
    stop = stops[0]
    assert all(b['t'] != 'Header' or (
        b['c'][0] == 2 and re.match(r'^(Riepilogo|Errata)(?:\s|$)', testo(b), re.I))
        for b in blocks[stop:]), 'Il riepilogo non è un finale autonomo.'
    if stop and blocks[stop - 1]['t'] == 'HorizontalRule':
        stop -= 1
    removed_labels = {b['c'][1][0] for b in blocks[stop:] if b['t'] == 'Header'}
    kept = list(range(stop))
    changes = []
    for i, b in enumerate(blocks[:stop]):
        if b['t'] == 'RawBlock' and '\\begin{indiceargomento}' in b['c'][1]:
            before = b['c'][1]
            b['c'][1] = '\n'.join(line for line in before.splitlines()
                                 if not any('[' + label + ']' in line for label in removed_labels))
            if b['c'][1] != before:
                changes.append({'indice_originale': i, 'tipo': 'indice locale aggiornato'})
        elif b['t'] == 'BlockQuote' and re.match(r'^Errata slide ', testo(b)):
            # Il titolo è un unico Strong. Il testo della nota è un altro blocco
            # e resta identico, comprese formulazione della fonte e correzione.
            title = b['c'][0]['c'][0]
            assert title['t'] == 'Strong'
            original = testo(b)
            assert title['c'][0] == {'t': 'Str', 'c': 'Errata'}
            number = re.match(r'^Errata slide (\S+)', original)[1]
            preposition = 'sulle' if re.search('[–-]', number) else 'sulla'
            title['c'][0]['c'] = 'Nota'
            title['c'][1:1] = [{'t': 'Space'}, {'t': 'Str', 'c': preposition}]
            changes.append({'indice_originale': i, 'tipo': 'sola etichetta della nota locale',
                            'prima': original, 'dopo': testo(b)})
    result['blocks'] = blocks[:stop]
    return result, kept, changes
