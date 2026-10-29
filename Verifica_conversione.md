# Verifica della dispensa di teoria

Conversione dei quattro Markdown teorici, a partire da MLP. Manuale introduttivo escluso per istruzione esplicita.

Gli originali sono conservati senza modifiche. La dispensa trasferisce i contenuti dei Markdown; non costituisce una nuova revisione scientifica delle slide o delle trascrizioni.

## Copertura file per file

| Fonte | Righe originali | Righe escluse | Blocchi trasferiti | Formule | Tabelle | Immagini e diagrammi |
|---|---:|---:|---:|---:|---:|---:|
| mlp_appunti_completi.md | 1092 | 13 | 391 | 496 | 13 | 27 |
| 03_CNN.md | 1312 | 15 | 445 | 580 | 7 | 66 |
| 04_RNN.md | 919 | 16 | 297 | 431 | 13 | 17 |
| 05_Transformer.md | 957 | 16 | 339 | 378 | 10 | 17 |

## Criterio di conservazione

- Ogni riga originale è classificata come conservata oppure esclusa con motivazione e impronta SHA-256.
- Ogni blocco trasferito ha un identificatore nel sorgente LaTeX, un’impronta del contenuto strutturato e una del frammento LaTeX.
- Le formule sono confrontate con quelle dei contenuti originali mantenuti, rispettando anche le occorrenze ripetute.
- L’emissione del capitolo completo è confrontata con quella ottenuta assemblando i singoli blocchi.
- Ogni immagine originale è copiata senza modifiche e verificata byte per byte; i diagrammi conservano anche il sorgente Mermaid.
- Le domande esplicative nel corpo e gli esempi svolti sono conservati. Non sono presenti sezioni autonome finali di domande di ripasso nei quattro file.
- Riepiloghi, errata e precisazioni matematiche restano inclusi. Nessuna deduplicazione dei contenuti teorici.
- Il rinvio CNN «Sezione 3.1» è aggiornato con un riferimento LaTeX alla corrispondente sezione.
- I rimandi al manuale introduttivo sono etichettati come riferimenti a un documento separato: il manuale non è incluso.

## Esclusioni editoriali

Sono esclusi soltanto i metadati YAML iniziali e le agende o convenzioni iniziali qui riportate. I titoli sono mantenuti come capitoli e sezioni, con numerazione uniforme.

### mlp_appunti_completi.md

- Riga 16: **Agenda (slide 2):** Introduzione → Classificatori lineari → Modellare un neurone → Reti neurali → Addestrare una DNN → Regolarizzazione.

### 03_CNN.md

- Riga 17: **Convenzione dei riferimenti:** i numeri «slide» indicano la posizione della pagina nel PDF a 93 pagine e il suffisso delle immagini. Il contatore stampato in basso può differire a causa delle pagine di animazione.

- Riga 19: **Agenda (slide 2):** Rappresentazione delle immagini → Paradigmi di Computer Vision → Introduzione ed evoluzione da MLP a CNN → Architettura dei blocchi convoluzionali (filtri LSI, connettività sparsa, condivisione dei pesi, feature maps, receptive field, iperparametri) → Visualizzazione dei filtri e gerarchia dei concetti → Funzioni di attivazione (ReLU, Leaky ReLU, ELU) → Pooling e classificatore → LeNet, AlexNet e VGG → ResNet → Batch Normalization fino all’inferenza.

### 04_RNN.md

- Riga 21: I riferimenti «slide N» indicano le pagine di `04_rnn.pdf`: in questo documento posizione nel PDF e numero stampato coincidono. La trattazione segue le slide fino alla 35; le slide 36–37 contengono bibliografia e ringraziamenti. Le parti presenti soltanto nelle slide e non sviluppate nella spiegazione sono segnalate con un callout informativo.

### 05_Transformer.md

- Riga 21: **Convenzione dei riferimenti:** «slide N» indica la **pagina fisica** di `Transformer.pdf` e il suffisso dell'immagine. Il PDF contiene 63 pagine ma il contatore stampato arriva a 44: diverse pagine sono animazioni della stessa slide. La parte spiegata termina con il confronto TCN–Transformer, alle pagine PDF 39–41, tutte numerate 32/44, e con un cenno orale alla cache di chiavi e valori. Le loss di regressione delle pagine successive sono rinviate e non vengono sviluppate qui.

## Dettaglio verificabile

Il registro JSON contiene la classificazione di tutte le righe, le trasformazioni editoriali, gli identificatori dei blocchi e le impronte dei file. I quattro JSON di contenuto conservano la struttura interpretata da Pandoc.

Compilazione: eseguire XeLaTeX due volte nella cartella che contiene il sorgente. Il file `.tex` contiene tutto il testo; la cartella `figure/` contiene i soli asset esterni necessari.


## Riorganizzazione per argomento

La dispensa è suddivisa in cinque argomenti autonomi: MLP, CNN, RNN, TCN e Transformer. TCN e Transformer, presenti nello stesso Markdown, hanno ora capitoli distinti. L’indice generale elenca soltanto questi argomenti; ogni capitolo contiene l’elenco navigabile dei propri sottoargomenti e apre su una nuova pagina. La regola è riportata anche nelle istruzioni redazionali per gli argomenti futuri.

Il corpo teorico è stato confrontato con la versione precedente ed è identico blocco per blocco. Sono cambiati i titoli principali, la numerazione conseguente e la navigazione. Il confronto indipendente del LaTeX conferma le 1.885 espressioni matematiche, le 43 tabelle e le 127 immagini o diagrammi; non risultano differenze nelle parole del testo normalizzate. I diagrammi mantengono gli stessi file già verificati.

Il PDF aggiornato conta 183 pagine. Sono stati controllati i cinque argomenti nell’indice, le aperture di capitolo, le tavole di contatto di tutte le pagine e le aree di testo e immagini. Nessun riferimento irrisolto, errore di compilazione o carattere mancante; restano tre avvisi di 0,123 punti nella medesima tabella, senza tagli visibili.
