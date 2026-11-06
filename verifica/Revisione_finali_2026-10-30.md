# Revisione editoriale del 30 ottobre 2026

Rimuovere i finali autonomi di riepilogo ed errata; conservare quanto espresso nelle fonti e le note locali; nessun altro cambiamento teorico.

## Perimetro e conservazione

I cinque Markdown di teoria, le slide, le trascrizioni, le immagini e lo standard Markdown originale non sono modificati. Il controllo riguarda questa revisione editoriale: non viene dichiarata una nuova lettura scientifica di tutte le fonti del corso. La lezione GNN conserva il limite alla slide stampata 28, pagina PDF 29.

Sono esclusi soltanto i finali autonomi di riepilogo/errata e i loro separatori. Le note locali mantengono il testo preesistente, comprese le espressioni delle fonti quando riportate, gli esempi e le correzioni. La sola etichetta «Errata slide» diventa «Nota sulla/sulle slide». Definizioni, formule, esempi svolti, derivazioni, tabelle didattiche nel corpo, immagini e rinvii restano invariati.

Le tabelle finali ripetevano contenuti già spiegati: il registro JSON associa ciascuna riga e ciascun punto di riepilogo ai blocchi del corpo. Non è stato necessario spostare tabelle finali intere. Questo riscontro semantico è distinto dal confronto automatico, esatto, dei blocchi conservati.

| Fonte | Blocchi del corpo identici | Voci di riepilogo riscontrate | Precisazioni riscontrate | Formule del corpo identiche |
|---|---:|---:|---:|---:|
| mlp_appunti_completi.md | 381 | 10 | 5 | 479 |
| 03_CNN.md | 439 | 21 | 10 | 523 |
| 04_RNN.md | 280 | 16 | 16 | 386 |
| 05_Transformer.md | 330 | 18 | 19 | 342 |
| 06_GNN.md | 223 | 15 | 15 | 248 |

## Note sulla trascrizione conservate fuori dal testo di studio

### 04_RNN.md

Le forme trascritte come «Adamart/Adamard», «EFO» e «bad propagation» sono ricondotte rispettivamente a Hadamard, ai gate i,f,o e alla backpropagation, coerentemente con formule e contesto. I frammenti non intelligibili della domanda sul confronto con altri regressori non consentono di ricostruire un metodo preciso e non sono trasformati in contenuti aggiuntivi.

### 05_Transformer.md

Le forme trascritte come «deletion», «casual», «GCN/DCN/TCM», «MootGate/MootHead attention», «Venus/batteries» e «graph count» sono ricondotte, dove il contesto lo rende univoco, a dilation, causal, TCN, Multi-Head Attention, values e Grad-CAM. Il passaggio dai termini corrotti alle nozioni corrette segue formule e figure, senza introdurre nuovi moduli.

## Verifiche ripetibili

Un solo vincolo di impaginazione, l’ambiente `samepage` attorno a `F03-B0050` e `F03-B0051`, tiene il titolo della nota alla slide 12 insieme al suo testo, senza modificare i blocchi né aggiungere contenuto.

La fotografia precedente è conservata in `verifica/prima_revisione_finali/`. `strumenti/revisiona_finali.py` ripete esclusivamente la revisione autorizzata, conserva gli identificatori e i frammenti LaTeX dei blocchi non modificati e ricostruisce il registro. `strumenti/finali_editoriali.py` applica lo stesso criterio anche nelle future conversioni. Il controllo indipendente `verifica_semantica.py` confronta il nuovo contenuto strutturato con il LaTeX. Compilazione e resa PDF sono verificate separatamente.

## Esito della compilazione e del PDF

PDF di 197 pagine, sei capitoli, GNN fino alla slide 28. Due passaggi XeLaTeX riusciti; nessun riferimento irrisolto o carattere mancante. Restano i soli tre avvisi preesistenti di 0,123 punti. Verifica indipendente del testo e delle formule: PASS per tutti e cinque i file. Controllo visivo delle pagine e delle note completato; nessun contenuto rilevato fuori area. Il titolo della nota alla slide 12 di RNN e il suo testo sono insieme a pagina fisica 118.
