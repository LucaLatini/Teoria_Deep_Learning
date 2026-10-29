# Dispensa di teoria — Deep Learning

La dispensa comprende, nell’ordine, MLP, CNN, RNN, TCN e Transformer. Il manuale introduttivo di Machine Learning è escluso come richiesto.

- `Dispensa_Deep_Learning.pdf`: documento pronto da leggere, con indice navigabile.
- `Dispensa_Deep_Learning.tex`: unico sorgente LaTeX, contenente tutto il testo.
- `figure/`: immagini originali e diagrammi necessari alla compilazione.
- `00_Istruzioni_Appunti_LaTeX.md`: istruzioni redazionali per mantenere la coerenza con gli appunti Markdown.
- `Verifica_conversione.md`: rapporto sintetico di copertura ed esclusioni.
- `verifica/`: registro completo, impronte dei blocchi e verifiche indipendenti.
- `originali/`: copie inalterate dei quattro Markdown usati come fonte e dello standard redazionale di riferimento.
- `strumenti/`: script impiegati per conversione e controllo.

## Compilazione

Mantenere il file `.tex` e la cartella `figure/` nella stessa directory. Aprire un terminale in quella directory ed eseguire due volte:

```sh
xelatex -interaction=nonstopmode -halt-on-error Dispensa_Deep_Learning.tex
```

Il compilatore è XeLaTeX. Sono utilizzati font TeX Gyre Pagella, TeX Gyre Heros e Latin Modern, distribuiti con TeX Live. Nel proprio editor selezionare `Dispensa_Deep_Learning.tex` come documento principale. Non è necessario installare Mermaid per compilare: i diagrammi PDF sono già inclusi.

## Verifiche

Le spiegazioni non sono state sintetizzate. Restano inclusi esempi numerici, derivazioni, riepiloghi, errata, tabelle, immagini e diagrammi. Metadati iniziali e agende ripetute sono esclusi; le convenzioni editoriali iniziali sono conservate nel rapporto. Il controllo riguarda la fedeltà ai quattro Markdown e la resa del PDF, non una nuova revisione scientifica di tutte le fonti.

Per ripetere il confronto indipendente fra contenuto strutturato e LaTeX, con Python e Pandoc installati:

```sh
python strumenti/verifica_semantica.py
```

Il rapporto tecnico conserva anche i casi in cui il lettore LaTeX di Pandoc interpreta diversamente numeri testuali o numerazioni. Le verifiche delle formule e delle parole sono registrate separatamente. Gli script di rigenerazione dei diagrammi richiedono Mermaid, Playwright e Chrome locali; i loro percorsi vanno configurati per il proprio ambiente.

## Struttura per argomento

L’indice generale contiene soltanto i cinque argomenti principali. Ogni capitolo apre su una nuova pagina e contiene il proprio elenco navigabile delle sezioni. TCN e Transformer sono capitoli separati; il corpo teorico preesistente è conservato integralmente.
