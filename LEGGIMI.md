# Dispensa di teoria — Deep Learning

La dispensa comprende, nell’ordine, MLP, CNN, RNN, TCN, Transformer e GNN. Il manuale introduttivo di Machine Learning è escluso come richiesto.

- `Dispensa_Deep_Learning.pdf`: documento pronto da leggere, con indice navigabile.
- `Dispensa_Deep_Learning.tex`: unico sorgente LaTeX, contenente tutto il testo.
- `figure/`: immagini originali e diagrammi necessari alla compilazione.
- `00_Istruzioni_Appunti_LaTeX.md`: istruzioni redazionali per mantenere la coerenza con gli appunti Markdown.
- `Verifica_conversione.md`: rapporto sintetico di copertura ed esclusioni.
- `verifica/`: registro completo, impronte dei blocchi e verifiche indipendenti.
- `originali/`: copie inalterate dei cinque Markdown usati come fonte e dello standard redazionale di riferimento.
- `strumenti/`: script impiegati per conversione e controllo.

## Compilazione

Mantenere il file `.tex` e la cartella `figure/` nella stessa directory. Aprire un terminale in quella directory ed eseguire due volte:

```sh
xelatex -interaction=nonstopmode -halt-on-error Dispensa_Deep_Learning.tex
```

Il compilatore è XeLaTeX. Sono utilizzati font TeX Gyre Pagella, TeX Gyre Heros e Latin Modern, distribuiti con TeX Live. Nel proprio editor selezionare `Dispensa_Deep_Learning.tex` come documento principale. Non è necessario installare Mermaid per compilare: i diagrammi PDF sono già inclusi.

## Verifiche

Le spiegazioni non sono state sintetizzate. Restano inclusi esempi numerici, derivazioni, tabelle didattiche, immagini, diagrammi e note locali sulle fonti. Su richiesta del 30/10/2026 sono rimossi i finali autonomi di riepilogo ed errata, dopo averne riscontrato le informazioni nel corpo. Il testo delle note locali è conservato; la sola etichetta «Errata slide» diventa «Nota sulla/sulle slide». Metadati iniziali e agende ripetute sono esclusi; le convenzioni editoriali iniziali sono conservate nel rapporto. Il controllo della revisione riguarda la conservazione dei contenuti del corpo e la resa del PDF, non una nuova revisione scientifica di tutte le fonti.

Per ripetere il confronto indipendente fra contenuto strutturato e LaTeX, con Python e Pandoc installati:

```sh
python strumenti/verifica_semantica.py
```

Il rapporto tecnico conserva anche i casi in cui il lettore LaTeX di Pandoc interpreta diversamente numeri testuali o numerazioni. Le verifiche delle formule e delle parole sono registrate separatamente. Gli script di rigenerazione dei diagrammi richiedono Mermaid, Playwright e Chrome locali; i loro percorsi vanno configurati per il proprio ambiente.

## Struttura per argomento

L’indice generale contiene soltanto i sei argomenti principali. Ogni capitolo apre su una nuova pagina e contiene il proprio elenco navigabile delle sezioni. TCN e Transformer sono capitoli separati; il corpo teorico preesistente è conservato integralmente.

La revisione dei soli finali è descritta in `verifica/Revisione_finali_2026-10-30.md`; la corrispondenza di ogni voce esclusa con il corpo è documentata in `verifica/revisione_finali_2026-10-30.json`. I blocchi precedenti sono conservati in `verifica/prima_revisione_finali/` per un confronto ripetibile.

## GNN: lezioni del 29 e 30 ottobre 2026

Il capitolo integra `06 GNN.pdf`, `29-10.txt` e `30-10.txt` fino alla slide numerata 56 inclusa, corrispondente alla pagina fisica 58 del PDF. La lezione del 30/10 riprende dalla slide 20 e sviluppa dalla slide 30 i compiti su grafi, il pooling, le ST-GCN, le GAT e GraphCast. È una redazione da fonti didattiche; i dettagli presenti soltanto nelle slide sono segnalati nel testo. La nota Markdown di lavorazione è inclusa in `originali/06_GNN.md`; il testo definitivo è incorporato nel sorgente `.tex` unico. I rapporti dedicati `verifica/Verifica_GNN_2026-10-29.md` e `verifica/Verifica_GNN_2026-10-30.md` distinguono le due stesure e documentano copertura, precisazioni e verifiche numeriche.

Lo script `revisiona_finali.py` conserva il procedimento storico della sola revisione dei finali e interrompe l’esecuzione quando rileva fonti aggiornate: non va usato per ripristinare la vecchia fotografia sopra le lezioni successive. Per ricompilare la versione corrente sono sufficienti il sorgente unico e le figure.
