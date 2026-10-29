---
tipo: istruzioni
formato_destinazione: LaTeX e PDF
aggiornato: 2026-10-29
standard_di_riferimento: 00_Istruzioni_Appunti.md
tags: [istruzioni, deep-learning, latex, appunti]
---

# Istruzioni per la redazione degli appunti in LaTeX

## Scopo e rapporto con i Markdown

Prima di lavorare, leggere questo documento e `00_Istruzioni_Appunti.md`. Quest’ultimo resta lo standard di riferimento per la scelta delle fonti, il metodo didattico, la completezza e la voce del testo. Questo documento ne applica le regole alla dispensa LaTeX e introduce le modifiche di formato richieste dall’utente.

Il cambio di formato non deve cambiare il modo di spiegare. Gli appunti LaTeX devono mantenere la stessa chiarezza, progressione, profondità delle spiegazioni e integrazione fra slide e trascrizioni degli appunti Markdown. Non trasformarli in un formulario, una sintesi delle slide o un trattato più accademico.

Non aggiungere sezioni di domande teoriche, quiz, autovalutazione, esercizi da svolgere, flashcard o domande di ripasso. Gli esempi già svolti e i problemi spiegati passo per passo restano parte del testo. Una domanda posta durante la lezione può fornire contenuto utile: integrarne spiegazione e risposta nel paragrafo pertinente, preferibilmente in forma espositiva, senza costruire una sezione di interrogazione.

Le istruzioni esplicite dell’utente prevalgono. Non cambiare queste regole né lo standard Markdown per iniziativa autonoma.

## 1. Distinguere l’incarico prima di iniziare

**Redazione o aggiornamento da fonti didattiche.** Leggere tutte le fonti disponibili della lezione: trascrizione, slide complete come PDF, figure, annotazioni manoscritte e appunti esistenti. Integrare i contenuti secondo la procedura delle sezioni successive.

**Conversione di appunti Markdown esistenti.** Trasferire integralmente i contenuti autorizzati, senza riassumerli o riscriverli automaticamente. Gli interventi ordinari riguardano sintassi, titoli, numerazione, collegamenti e impaginazione. Una conversione fedele non equivale a una nuova verifica scientifica di tutte le fonti: non dichiarare di aver controllato slide o trascrizioni che non sono state lette. Segnalare separatamente gli eventuali problemi rilevati; registrare ogni correzione sostanziale.

Lavorare un file alla volta, conservarne un checkpoint verificato e procedere poi al successivo. Il numero di pagine o di parole non costituisce da solo una prova di completezza.

## 2. Fonti, gerarchia e perimetro

La gerarchia per ricostruire la spiegazione del corso rimane **trascrizione del professore > slide > appunti esistenti**.

- Le **slide sono lo scheletro**: determinano ordine, perimetro degli argomenti, definizioni, grafici e formule.
- La **trascrizione è la sostanza**: aggiunge i perché, i passaggi sottintesi, le analogie, i calcoli e i chiarimenti alle domande d’aula.
- Gli **appunti esistenti** mantengono la continuità terminologica e didattica. Non sostituiscono le fonti quando è richiesta una nuova redazione.

La gerarchia ricostruisce quanto spiegato; non autorizza a riprodurre un errore matematico o una parola trascritta male. Verificare i passaggi e registrare le correzioni con il riferimento alla fonte. Se l’audio o il testo sono ambigui e il contesto non basta, dichiarare l’incertezza senza inventare la spiegazione.

Non aggiungere teoremi collaterali, dimostrazioni astratte, approfondimenti specialistici o argomenti non trattati. Sono ammesse brevi precisazioni indispensabili alla correttezza, riconoscibili come tali. Non usare fonti web per ampliare il programma senza una richiesta pertinente.

Distinguere ciò che è stato spiegato da ciò che compare soltanto nelle slide. Se manca una fonte, registrarlo nel rapporto di lavoro; non presentare il controllo delle fonti come completo.

Per la dispensa attuale il percorso autorizzato è **MLP → CNN → RNN → TCN → Transformer**. `Manuale_Studio_Machine_Learning_to_Deep_Learning.md` resta escluso finché l’utente non ne richiede esplicitamente l’inclusione. I richiami a quel documento non autorizzano a importarne i capitoli. I laboratori non fanno parte della dispensa teorica attuale.

## 3. Struttura e continuità della spiegazione

### Gerarchia obbligatoria: un argomento principale per capitolo

L’unità organizzativa è l’argomento, non il file sorgente o la data di lezione. Ogni argomento principale deve avere una propria sezione autonoma, realizzata come capitolo LaTeX, con apertura su una nuova pagina. Non riunire due architetture distinte nello stesso capitolo soltanto perché provengono dallo stesso Markdown o PDF.

| Livello principale | Contenuti subordinati |
|---|---|
| MLP | Neurone, attivazioni, struttura della rete, loss, backpropagation, inizializzazione e regolarizzazione. |
| CNN | Immagini, convoluzioni, canali, campo recettivo, stride e padding, pooling, architetture convoluzionali, residui e BatchNorm. |
| RNN | Serie temporali, ricorrenza, hidden state, unfolding, BPTT e architetture ricorrenti con gate, entro il programma svolto. |
| TCN | Convoluzione 1D, dilatazione, causalità, campo recettivo temporale e blocchi convoluzionali per sequenze. |
| Transformer | Token, embedding e posizione, encoder, attention, Q/K/V, multi-head, MLP, LayerNorm e aggregazione finale. |

L’indice generale deve mostrare soltanto gli argomenti principali, allo stesso livello. Ogni capitolo deve aprirsi con un elenco navigabile dei propri sottoargomenti. Sezioni e sottosezioni restano all’interno dell’argomento a cui appartengono, con numerazione e riferimenti coerenti.

TCN e Transformer devono quindi avere capitoli distinti anche quando si trovano nello stesso file sorgente. I paragrafi di confronto o raccordo restano conservati nel punto pertinente: separare la struttura non significa cancellare i collegamenti didattici. Per qualunque nuovo argomento principale richiesto in futuro, aggiungere un nuovo capitolo allo stesso livello; non accumularlo nell’ultimo capitolo già presente. Un approfondimento dello stesso argomento resta invece una sezione o sottosezione del relativo capitolo.


Seguire l’ordine delle slide. Mantenere i riferimenti alle slide nei titoli delle sezioni, distinguendo pagina fisica del PDF e numero stampato quando non coincidono. Non inventare numeri o intervalli.

Costruire un discorso omogeneo, come in un manuale. Fondere contenuto delle slide e spiegazione orale nel punto in cui servono. Non usare formule quali «il docente sottolinea», «il professore dice», «come visto a lezione» o riquadri che separino la voce del professore dal resto del testo.

Conservare analogie e casi concreti del corso, anche quando sembrano informali: aiutano a comprendere e ricordare. Mantenere la notazione già introdotta, dichiarando esplicitamente eventuali cambi di convenzione, orientamento dei vettori o assi dei tensori.

Organizzare il documento con un solo titolo generale, un indice navigabile e capitoli e sezioni coerenti. Non ripetere all’inizio di ogni capitolo frontmatter, agenda, prerequisiti, date e inventario delle fonti. Conservare questi dati nei commenti del sorgente o nel registro separato. I titoli degli argomenti restano presenti come titoli di capitolo o sezione.

I raccordi fra capitoli devono chiarire un collegamento già sostenuto dai materiali. Non aggiungere un argomento soltanto per rendere più elegante la transizione.

## 4. Stile didattico identico allo standard Markdown

Scrivere in italiano, con tono limpido, diretto e pragmatico. L’obiettivo è rendere lo studio e l’esposizione d’esame più comprensibili.

- Preferire spiegazioni collegate e paragrafi fluidi. Usare elenchi per passaggi, proprietà o confronti realmente paralleli, non per frammentare ogni frase.
- Spiegare il termine tecnico quando viene introdotto e mantenere poi la stessa terminologia.
- Evitare gergo barocco, enfasi retorica, commenti motivazionali e riempitivi enciclopedici.
- Non sostituire un’intuizione chiara con una formulazione più astratta soltanto perché la destinazione è LaTeX.
- Conservare gli esempi numerici e le analogie utili del corso; verificare i numeri.
- Non eliminare automaticamente una ripetizione didattica: può introdurre un concetto da una prospettiva diversa.

**Regola aurea:** ogni riga aggiunta deve semplificare comprensione ed esposizione, senza aggiungere nozionismo sterile o pesantezza retorica.

## 5. Formule, derivazioni e teoremi

Per ogni formula, derivazione o teorema importante conservare i cinque elementi dello standard Markdown:

1. **Cosa dice.** Formula ben distinta, eventualmente con `\boxed{...}`, e significato di ogni simbolo. Esplicitare dimensioni e forme di vettori, matrici e tensori.
2. **Da dove viene.** Derivazione completa, un passaggio per riga, dichiarando la regola utilizzata. Non saltare i passaggi che consentono di capire il risultato.
3. **Intuizione.** Significato in parole, con le spiegazioni e le analogie dei materiali integrate nel testo.
4. **Esempio numerico.** Esempio piccolo, svolto e verificato a mano; tabella quando aiuta a leggere il calcolo.
5. **Collegamenti.** Rinvii alle sezioni pertinenti tramite etichette e riferimenti LaTeX stabili.

Durante una nuova redazione, costruire questi elementi entro il perimetro del corso. Durante una conversione, trasferire quelli presenti senza inventare esempi o derivazioni per colmare automaticamente una lacuna.

Usare ambienti matematici appropriati, come `aligned`, per rendere visibili i passaggi. Non impaginare una formula come immagine se è disponibile la sua espressione matematica. Per una formula troppo lunga, preferire una divisione leggibile a un ridimensionamento che la renda minuscola. Non cambiare simboli o convenzioni per risolvere un problema di spazio.

## 6. Immagini, diagrammi, tabelle e codice

Conservare le figure utili e verificare l’esistenza dei file collegati. Usare la slide intera quando contiene annotazioni manoscritte rilevanti; altrimenti è possibile usare il ritaglio della figura, senza perdere etichette, assi, legenda e contesto necessario.

I diagrammi Mermaid devono diventare figure effettivamente leggibili nel PDF. Conservare anche il sorgente del diagramma. Si può adattare l’orientamento alla pagina mantenendo nodi, etichette, collegamenti e significato.

Le tabelle devono conservare intestazioni, righe, colonne, unità, simboli e valori. Usare tabelle su più pagine quando necessario, con intestazioni ripetute. Verificare che le formule nelle celle non siano tagliate.

Conservare il codice necessario alla spiegazione teorica. Se viene richiesto un documento di laboratorio separato, mantenere la struttura **Teoria → Notebook cella per cella → Errori ed esperimenti**, collegare il codice alle formule, includere le forme dei tensori e interpretare i risultati rispetto a una baseline. Non correggere autonomamente un notebook assegnato come esercizio.

## 7. Cosa conservare e cosa escludere

| Elemento dei Markdown | Regola per la dispensa LaTeX |
|---|---|
| Spiegazioni, definizioni e derivazioni | Conservare integralmente; stessa progressione didattica. |
| Esempi svolti e calcoli a mano | Conservare con tutti i passaggi e i risultati. |
| Analogie e chiarimenti emersi durante la lezione | Integrare nel testo, senza attribuzioni separate. |
| Domande di ripasso, quiz e autovalutazione | Non aggiungere; se presenti nella fonte, escludere la forma interrogativa finale e recuperare nel capitolo le informazioni teoriche uniche delle risposte. |
| Domande che introducono una spiegazione nel corpo | Conservare il contenuto esplicativo; nella nuova redazione preferire una formulazione espositiva. |
| Riepiloghi di formule e concetti | Conservare: non sono domande di ripasso. |
| Errata, avvisi, precisazioni e teoremi | Conservare con evidenza sobria e riferimento alla fonte. |
| Contenuti soltanto nelle slide | Conservare con una nota informativa che ne specifichi lo stato. |
| Frontmatter e intestazioni editoriali ripetute | Trasferire in commenti o registro separato; non ripeterli nella dispensa. |
| Titoli e numerazione | Mantenere gli argomenti e uniformare la numerazione; aggiornare i rinvii. |
| Wikilink | Convertire in riferimenti interni o in richiami leggibili a documenti separati. |
| Figure, diagrammi e tabelle | Conservare e verificarne la resa nel PDF. |

Non eliminare informazioni soltanto perché si trovano dentro un callout o in una risposta. Non deduplicare, sintetizzare o accorciare il contenuto teorico senza un’istruzione esplicita.

## 8. Continuità del sorgente e dell’impaginazione

Usare la dispensa esistente come riferimento visivo: preambolo, caratteri, colori, margini, gerarchia dei titoli e stile delle figure. Non ridisegnare l’impaginazione a ogni nuova lezione.

Mantenere un unico sorgente principale `.tex` contenente tutto il testo. Le immagini possono restare in una cartella esterna con percorsi relativi. Il PDF resta un file unico. Se vengono usati frammenti durante il lavoro, assemblarli nel sorgente finale e conservare la tracciabilità nel rapporto separato.

Non lasciare sintassi Obsidian visibile, wikilink irrisolti, metadati tecnici o segnaposto nel PDF. Usare etichette univoche e rinvii aggiornati; non importare il contenuto di una nota esterna per il solo fatto che esiste un collegamento.

## 9. Verifica prima della consegna

Per ogni file, classificare ogni blocco come trasferito oppure escluso con un motivo coerente con le istruzioni. Conservare un registro separato: non appesantire la dispensa con il rapporto tecnico.

Controllare almeno:

- copertura di paragrafi, formule, esempi, tabelle, figure e diagrammi, compresi i callout;
- corrispondenza delle formule e dei valori, con le occorrenze ripetute quando presenti;
- conservazione delle informazioni uniche prima di escludere eventuali domande finali;
- correttezza dei rinvii e distinzione fra numero di slide e pagina PDF;
- stato delle fonti lette e delle parti non ancora spiegate;
- compilazione del sorgente e assenza di riferimenti irrisolti, caratteri mancanti o contenuti tagliati;
- leggibilità del PDF mediante rendering e controllo visivo, soprattutto per formule, tabelle e figure;
- integrità degli originali e coerenza fra sorgente consegnato e PDF.

Una compilazione riuscita non dimostra da sola che il contenuto sia completo. Una corrispondenza testuale non dimostra da sola la leggibilità del PDF. Una conversione fedele non certifica la correttezza scientifica di tutte le affermazioni della fonte. Descrivere le verifiche realmente svolte, senza promesse assolute non sostenute dai controlli.

## 10. Salvataggi e continuità del lavoro

Prima di sovrascrivere una nota, salvarne una copia in `_archivio/`, come previsto dallo standard originale. Conservare i Markdown sorgente durante le conversioni. Salvare un checkpoint dopo ogni file verificato e annotare ciò che resta da completare.

Dopo aver aggiunto o aggiornato un documento, aggiornare `00_Indice_Corso.md`. Conservare con la dispensa un rapporto sintetico delle esclusioni, delle correzioni sostanziali e delle limitazioni note. Le istruzioni redazionali e il rapporto di verifica restano documenti separati dal testo da studiare.

## Incarico standard da applicare nelle sessioni successive

Leggere `00_Istruzioni_Appunti.md` e questo documento; identificare se si tratta di nuova redazione o conversione; definire l’insieme dei file autorizzati; lavorare in ordine, un file alla volta; mantenere fonti, metodo e stile dei Markdown; conservare le spiegazioni complete; non aggiungere domande di teoria o ripasso; verificare la corrispondenza dei contenuti e la leggibilità del PDF prima della consegna.
