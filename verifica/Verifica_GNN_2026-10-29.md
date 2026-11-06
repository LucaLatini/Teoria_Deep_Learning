> Rapporto della stesura del 29/10/2026, precedente alla revisione editoriale del 30/10. I conteggi della nota sorgente e le pagine qui riportati descrivono quella versione; per il PDF aggiornato vedere `Revisione_finali_2026-10-30.md` e `controllo_pdf.json`.

# Verifica della lezione GNN del 29 ottobre 2026

## Perimetro e modalità di lavoro

Questo capitolo è una nuova redazione basata su `03_Fonti/Slide/06 GNN.pdf` e sulla trascrizione completa `03_Fonti/Trascrizioni/29-10.txt`. Segue `00_Istruzioni_Appunti.md` e `00_Istruzioni_Appunti_LaTeX.md`. La nota di lavorazione `01_Teoria/06_GNN.md` confluisce integralmente nel sorgente LaTeX unico, come sesto argomento autonomo.

Il limite è la **slide stampata 28 inclusa**, corrispondente alla **pagina fisica 29**. La slide stampata 19 occupa due pagine successive (19 e 20): sono state lette entrambe. Le applicazioni, GAT e GraphCast che seguono nel PDF non sono sviluppati. Le relative voci nell’agenda non sono trattate come contenuti già spiegati.

I primi quattro file di appunti, che formano i cinque argomenti preesistenti, mantengono gli stessi frammenti di testo. L’aggiunta non importa il manuale introduttivo e non modifica i notebook di laboratorio.

## Copertura delle slide

| Slide stampate | Pagine PDF | Destinazione nella nota GNN |
|---|---|---|
| 1-3 | 1-3 | Titolo, agenda e separatore: usati per il perimetro, senza ripetere intestazioni nel testo. |
| 4-5 | 4-5 | Sezione 1: CNN, supporto del filtro, formula della correlazione ed esempio numerico. |
| 6-7 | 6-7 | Sezione 1.2: reticoli 2D e 1D; figure estratte dai materiali. |
| 8-9 | 8-9 | Sezione 2: grafo, nodi, archi e feature; reticolo come caso particolare. |
| 10-11 | 10-11 | Sezione 3.1: limiti della griglia regolare ed esempi di domini irregolari. |
| 12 | 12 | Separatore dell’argomento: non aggiunge una spiegazione teorica autonoma. |
| 13-14 | 13-14 | Sezioni 3.2-3.3: equivarianza/invarianza, vicinati, layout e topologia. |
| 15 | 15 | Sezione 4: GNN e specifica GCN, livelli, parametri e addestramento. |
| 16-17 | 16-17 | Sezione 4: aggregazione, trasformazione e aggiornamenti coerenti dello strato. |
| 18 | 18 | Sezione 5.1: X e A, forme e distinzione fra l’esempio orientato della figura e l’ipotesi non orientata della derivazione. |
| 19 | 19-20 | Sezione 5.2: prior strutturale, posa, dominio, k-NN e soglie. |
| 20 | 21 | Sezione 6.1: self-loop, gradi e normalizzazione simmetrica. |
| 21 | 22 | Sezione 6.2: proiezione, dimensioni, parametri e attivazione. |
| 22 | 23 | Sezione 7: forma per nodo, pesi degli archi, associatività ed esempio numerico completo. |
| 23 | 24 | Sezione 8: campo recettivo e formula a due livelli; collegamento con la motivazione spettrale. |
| 24 | 25 | Sezione 9: Laplaciano, espansione per nodo, autovettori e polinomi. |
| 25 | 26 | Sezione 10: base ortonormale, trasformata e inversa, filtro spettrale ed esempio a due nodi. |
| 26 | 27 | Sezione 11: ricorrenza di Chebyshev, riscalatura, coefficienti e grado. |
| 27 | 28 | Sezione 12: cancellazione di U, identità polinomiale e ricorrenza diretta sui segnali. |
| 28 | 29 | Sezione 13: primo grado, vincolo sui coefficienti, rinormalizzazione e GCN multifeature. |
| 29-60 | 30-62 | Fuori dal termine richiesto; non sviluppate. |

Le sezioni 14-15 raccolgono le formule già spiegate e le precisazioni necessarie. Non contengono domande di ripasso.

## Integrazione della trascrizione

La trascrizione è stata letta integralmente in tre intervalli consecutivi di caratteri: 0-17500, 17500-35000, 35000-fine. Sono stati integrati i passaggi relativi a:

- griglia regolare, kernel, reticolo delle immagini e asse temporale;
- acqua, strade, telefonate, citazioni, mesh e articolazioni come esempi di struttura;
- rinumerazione dei nodi e distinzione rispetto allo spostamento fisico dei dati;
- messaggi di primo e secondo ordine, dipendenze e componenti disconnesse;
- feature, adiacenza, gradi, proiezione, possibilità di precomputare la normalizzazione;
- associatività e costo relativo dell’ordine dei prodotti;
- richiamo al teorema della convoluzione nel dominio di Fourier;
- autovalori, autovettori, cambio di coordinate e analogia con la PCA;
- polinomi della matrice e dei suoi autovalori, riscalatura e ricorrenza di Chebyshev;
- filtro di primo grado, coefficienti opposti, controllo della scala e formula finale.

Ripetizioni orali sono fuse nella spiegazione pertinente; richieste di conferma alla classe, esitazioni e comunicazioni organizzative finali sono escluse. Non sono trasformate in domande di ripasso. L’ultima parte della trascrizione conferma che le applicazioni e le architetture ulteriori sono rinviate.

I termini trascritti in forma corrotta, fra cui le varianti di «adiacenza», «Laplaciano», «Kipf e Welling», «Chebyshev», «VGG», «MLP» ed «eigenvalues/eigenvectors», sono ricostruiti usando formule e contesto delle slide; le parole incomprensibili non danno origine a nuovi argomenti.

## Precisazioni matematiche ed editoriali

- La distinzione fra convoluzione e correlazione riguarda il ribaltamento del kernel.
- La rete con una riga di uscita per nodo è equivarante alla rinumerazione; l’invarianza globale e quella alle rotazioni sono proprietà differenti.
- Il prodotto normalizzato usato per propagare i messaggi è un’adiacenza normalizzata, non il Laplaciano.
- La dimensione dell’ingresso nella slide 21 è corretta coerentemente con la matrice W e lo strato precedente.
- Il grafo orientato nella figura della slide 18 non viene usato come giustificazione di una diagonalizzazione simmetrica: l’ipotesi per la derivazione è esplicita.
- Il numeratore del messaggio va mantenuto per archi pesati; la normalizzazione simmetrica non è una media per riga.
- La scomposizione simmetrica agli autovalori è distinta dalla SVD generale.
- L’identità polinomiale vale come identità matriciale su qualsiasi segnale; la scelta di un filtro polinomiale può invece essere approssimata.
- I termini T0 e T1 definiscono il grado 1; il vincolo sui coefficienti è una scelta di modello.
- La rinormalizzazione cambia l’operatore: I+S non è uguale all’adiacenza rinormalizzata. La limitazione della sola propagazione non garantisce stabilità di tutta la rete.

Gli esempi numerici piccoli sono ricostruzioni didattiche entro le operazioni trattate, non risultati sperimentali attribuiti alla lezione. Sono verificati nel file `verifica_numerica_GNN.json`: GCN su tre nodi, ReLU, associatività, permutazione coerente, base spettrale su due nodi, trasformata e ricostruzione, filtro, ricorrenza e rinormalizzazione.

## Controlli documentali

`fonti_GNN.json` conserva impronte dei due input e corrispondenza esatta fra pagine PDF e slide stampate. Il registro generale conserva righe, blocchi e formule della nota. Il controllo indipendente rilegge il LaTeX effettivamente emesso e verifica testo, formule, tabelle e immagini. Il controllo del PDF, della navigazione e dell’impaginazione è registrato dopo la compilazione finale.
