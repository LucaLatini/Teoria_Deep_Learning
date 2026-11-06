# Verifica della dispensa di teoria

Dispensa per argomenti MLP, CNN, RNN, TCN, Transformer e GNN. Primi quattro Markdown conservati nella conversione preesistente. GNN integra le lezioni del 29 e 30/10/2026 fino alla slide stampata 56 (pagina PDF 58), con ripasso dalle slide 20-22 e nuova parte dalle slide 30-56. Manuale introduttivo escluso. Nessun riepilogo o errata finale.

I quattro Markdown precedenti a GNN sono conservati senza modifiche: il loro trasferimento non costituisce una nuova revisione scientifica. GNN è una redazione da slide e trascrizioni, aggiornata con le lezioni richieste; i relativi rapporti distinguono la verifica delle fonti dalla conversione. Il Markdown di lavorazione corrente è conservato fra gli originali della conversione e le versioni precedenti sono archiviate.

## Copertura file per file

| Fonte | Righe originali | Righe escluse in preparazione | Blocchi mantenuti | Formule | Tabelle | Immagini e diagrammi |
|---|---:|---:|---:|---:|---:|---:|
| mlp_appunti_completi.md | 1092 | 13 | 386 | 479 | 13 | 27 |
| 03_CNN.md | 1312 | 15 | 440 | 523 | 5 | 66 |
| 04_RNN.md | 919 | 16 | 290 | 386 | 11 | 17 |
| 05_Transformer.md | 957 | 16 | 332 | 342 | 8 | 17 |
| 06_GNN.md | 1071 | 11 | 439 | 473 | 5 | 21 |

## Criterio di conservazione

- Ogni riga originale è classificata nella preparazione iniziale come conservata oppure esclusa con motivazione e impronta SHA-256. Le esclusioni editoriali successive sono registrate separatamente per blocco.
- Ogni blocco trasferito ha un identificatore nel sorgente LaTeX, un’impronta del contenuto strutturato e una del frammento LaTeX.
- Le formule sono confrontate con quelle dei contenuti mantenuti nel corpo, rispettando anche le occorrenze ripetute. Le occorrenze nei soli finali esclusi sono documentate nel registro della revisione.
- L’emissione del capitolo completo è confrontata con quella ottenuta assemblando i singoli blocchi.
- Ogni immagine originale è copiata senza modifiche e verificata byte per byte; i diagrammi conservano anche il sorgente Mermaid.
- Le spiegazioni nel corpo e gli esempi svolti sono conservati. Non sono aggiunte sezioni autonome finali di domande di ripasso.
- Su richiesta del 30/10/2026 sono rimossi i finali autonomi di riepilogo ed errata. Le note locali, i riferimenti alla formulazione delle fonti e le precisazioni restano accanto ai passaggi pertinenti. Il registro della revisione riscontra nel corpo ogni voce dei finali esclusi.
- Il rinvio CNN «Sezione 3.1» è aggiornato con un riferimento LaTeX alla corrispondente sezione.
- I rimandi al manuale introduttivo sono etichettati come riferimenti a un documento separato: il manuale non è incluso.

## Esclusioni editoriali

Nella preparazione iniziale sono esclusi i metadati YAML e le agende o convenzioni iniziali qui riportate. La successiva revisione del 30/10/2026 esclude soltanto i finali autorizzati di riepilogo ed errata, con riscontro delle informazioni nel corpo e conservazione delle note locali. I titoli restano capitoli e sezioni con numerazione uniforme.

### mlp_appunti_completi.md

- Riga 16: **Agenda (slide 2):** Introduzione → Classificatori lineari → Modellare un neurone → Reti neurali → Addestrare una DNN → Regolarizzazione.

### 03_CNN.md

- Riga 17: **Convenzione dei riferimenti:** i numeri «slide» indicano la posizione della pagina nel PDF a 93 pagine e il suffisso delle immagini. Il contatore stampato in basso può differire a causa delle pagine di animazione.

- Riga 19: **Agenda (slide 2):** Rappresentazione delle immagini → Paradigmi di Computer Vision → Introduzione ed evoluzione da MLP a CNN → Architettura dei blocchi convoluzionali (filtri LSI, connettività sparsa, condivisione dei pesi, feature maps, receptive field, iperparametri) → Visualizzazione dei filtri e gerarchia dei concetti → Funzioni di attivazione (ReLU, Leaky ReLU, ELU) → Pooling e classificatore → LeNet, AlexNet e VGG → ResNet → Batch Normalization fino all’inferenza.

### 04_RNN.md

- Riga 21: I riferimenti «slide N» indicano le pagine di `04_rnn.pdf`: in questo documento posizione nel PDF e numero stampato coincidono. La trattazione segue le slide fino alla 35; le slide 36–37 contengono bibliografia e ringraziamenti. Le parti presenti soltanto nelle slide e non sviluppate nella spiegazione sono segnalate con un callout informativo.

### 05_Transformer.md

- Riga 21: **Convenzione dei riferimenti:** «slide N» indica la **pagina fisica** di `Transformer.pdf` e il suffisso dell'immagine. Il PDF contiene 63 pagine ma il contatore stampato arriva a 44: diverse pagine sono animazioni della stessa slide. La parte spiegata termina con il confronto TCN–Transformer, alle pagine PDF 39–41, tutte numerate 32/44, e con un cenno orale alla cache di chiavi e valori. Le loss di regressione delle pagine successive sono rinviate e non vengono sviluppate qui.

### 06_GNN.md

## Criterio editoriale dei finali, concordato il 30 ottobre 2026

La revisione storica dei soli finali ha lasciato invariati i Markdown allora presenti; le successive stesure da nuove lezioni sono documentate separatamente. Nella destinazione LaTeX i titoli delle note locali passano da «Errata slide» a «Nota sulla/sulle slide» senza riscrivere il testo delle note. Gli indici locali non elencano i finali esclusi e i nuovi argomenti non introducono riepiloghi o errata autonomi.

Il riscontro per voce e la verifica esatta dei blocchi sono documentati in `verifica/Revisione_finali_2026-10-30.md` e `verifica/revisione_finali_2026-10-30.json`. La situazione precedente è conservata in `verifica/prima_revisione_finali/`. Le note sui termini corrotti nella trascrizione sono trasferite al rapporto tecnico, mentre i termini corretti restano nel corpo.

## Dettaglio verificabile

Il registro JSON contiene la classificazione di tutte le righe, le trasformazioni editoriali, gli identificatori dei blocchi e le impronte dei file. I JSON di contenuto conservano la struttura interpretata da Pandoc.

Compilazione: eseguire XeLaTeX due volte nella cartella che contiene il sorgente. Il file `.tex` contiene tutto il testo; la cartella `figure/` contiene i soli asset esterni necessari.

## Lezione GNN del 30 ottobre 2026

Il capitolo GNN integra le trascrizioni del 29 e 30/10 e le slide fino alla slide numerata 56, pagina fisica 58. La nuova lezione riprende dalle slide 20-22 e sviluppa dalla slide 30 classificazione di nodi e grafi, link prediction, supervisione transduttiva/induttiva, pooling, ST-GCN, GAT, multi-head e GraphCast. I dettagli di interpretabilità e GATv2 presenti soltanto nelle slide sono distinti dalla spiegazione orale. Non sono introdotti finali autonomi di riepilogo o errata.

Il PDF finale conta 217 pagine; il capitolo GNN occupa le pagine fisiche 177-217. Le prime 176 pagine e i primi quattro frammenti sono invariati rispetto alla versione precedente. Tutte le 248 espressioni matematiche della prima parte GNN e i suoi riferimenti sono conservati. Sono aggiornate due sole frasi di rinvio e la figura della slide 20, ora intera con le annotazioni del ripasso.

Il contenuto GNN corrente comprende 473 espressioni matematiche, 5 tabelle didattiche e 21 immagini. Il confronto indipendente LaTeX restituisce PASS per tutti i file, senza differenze nelle formule, parole normalizzate, tabelle e immagini rispetto ai contenuti mantenuti. I nuovi esempi numerici sono verificati indipendentemente. Due passaggi XeLaTeX riusciti; nessun nuovo avviso, errore, carattere mancante o riferimento irrisolto. Restano i soli tre avvisi preesistenti di 0,123 punti nella stessa tabella.

Sono state renderizzate e ispezionate le 41 pagine GNN, con controlli di dettaglio e una nuova verifica dopo l'allineamento dei passaggi matematici. Nessun contenuto rilevato fuori dall'area verificata. Copertura, precisazioni e provenienza sono documentate in `verifica/Verifica_GNN_2026-10-30.md`, `fonti_GNN_30-10.json`, `conservazione_GNN_30-10.json` e `verifica_numerica_GNN_30-10.json`.
