"""Riordina la gerarchia senza riconvertire o modificare il corpo degli appunti."""
import copy, importlib.util, json
from pathlib import Path
spec=importlib.util.spec_from_file_location('converti',Path(__file__).with_name('converti.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.REPORT=json.loads((m.OUT/'verifica/registro_conversione.json').read_text('utf-8'))
fragments=[]
for ordinal,record in enumerate(m.REPORT['fonti'],1):
    path=m.OUT/f'verifica/f{ordinal:02d}_contenuto.json'
    old=json.loads(path.read_text('utf-8'))
    new=m.organizza_argomenti(copy.deepcopy(old),ordinal)
    def body(ast):
        return [b for b in ast['blocks'] if b['t']!='Header' and not (b['t']=='RawBlock' and '\\begin{indiceargomento}' in b['c'][1])]
    assert body(old)==body(new), 'Il corpo teorico è stato alterato'
    assert [n['c'] for n in m.walk(old) if n['t']=='Math']==[n['c'] for n in m.walk(new) if n['t']=='Math']
    blocktex=[];records=[]
    for j,block in enumerate(new['blocks'],1):
        ident=f'F{ordinal:02d}-B{j:04d}'
        one={'pandoc-api-version':new['pandoc-api-version'],'meta':{},'blocks':[block]}
        piece=m.tex_fragment(one)
        blocktex.append(f'% BEGIN {ident}\n'+piece+f'% END {ident}\n')
        records.append({'id':ident,'tipo':block['t'],'sha256_ast':m.sha(json.dumps(block,sort_keys=True,ensure_ascii=False)),'sha256_tex':m.sha(piece),'conteggi':m.counts(block)})
    fragment='\n'.join(blocktex)
    (m.BUILD/f'f{ordinal:02d}.tex').write_text(fragment,encoding='utf-8')
    (m.OUT/f'verifica/f{ordinal:02d}_frammento.txt').write_text(fragment,encoding='utf-8')
    m.write_json(path,new)
    record['blocchi']=records;record['conteggi_finali']=m.counts(new);record['sha256_frammento_tex']=m.sha(fragment)
    record['controllo_corpo_dopo_riorganizzazione']='identico blocco per blocco alla versione precedente; sole modifiche a titoli e navigazione'
    fragments.append(fragment)
    print('Riorganizzato',record['file'],flush=True)
m.checkpoint(fragments);m.audit()
