---
tipo: istruzioni
aggiornato: 2026-10-16
tags: [istruzioni, deep-learning]
---

# Istruzioni per la stesura degli appunti

Regole da seguire per scrivere, revisionare o integrare gli appunti di questo vault, sia per le lezioni di teoria sia per quelle di laboratorio. Valgono per chiunque lavori sugli appunti, persona o assistente AI. Sono le stesse regole della skill `appunti-deep-learning` di Claude.

## Dove sono i file

- `00_Indice_Corso.md`: l'indice, da aggiornare dopo ogni nuova nota
- `01_Teoria/`: una nota per ogni argomento di teoria
- `02_Laboratorio/`: le note dei laboratori
- `03_Fonti/`:
  - `Slide/`: i PDF delle slide
  - `Trascrizioni/`: le trascrizioni delle lezioni, chiamate `AAAA-MM-GG_argomento.txt`
  - `Testo_slide/`: il testo estratto dalle slide
- `04_Notebook_Lab/`: i notebook con i relativi dati
- `assets/`: le immagini, chiamate `<argomento>_slide_NN.png`
- `_archivio/`: le versioni precedenti delle note

## Procedura

1. Leggi **tutte** le fonti della lezione:
   - la trascrizione;
   - le slide, guardando anche il PDF per figure e annotazioni a mano;
   - la nota esistente.
2. La gerarchia delle fonti è: **trascrizione del prof > slide > appunti esistenti**.
3. Usa le slide come scheletro: le sezioni seguono il loro ordine e il titolo riporta il numero di slide.
4. **Scrivi un testo omogeneo, senza attribuzioni.** Le spiegazioni, gli esempi, le analogie e le domande fatte a voce dal prof si fondono con il testo della slide in un unico discorso continuo, come in un manuale.
   - Non usare riquadri `> [!quote] Il docente`.
   - Non usare formule come "il docente sottolinea", "il prof dice", "a lezione".
   - Il contenuto orale entra nel punto della spiegazione in cui serve, con lo stesso tono del resto.
5. Segnala con `> [!info]` solo le parti che sono nelle slide ma non sono state ancora spiegate a lezione.
6. Verifica i conti e le formule delle slide. Correggi gli errori con `> [!bug] Errata slide N` e riepilogali in fondo alla nota.
7. Non aggiungere argomenti non trattati, salvo brevi precisazioni necessarie alla correttezza.
8. Attenzione alle parole trascritte per errore o trascritte male, porre attenzione a questo aspetto.

## Calibrazione del tono: Efficacia didattica vs Iper-accademicizzazione artificiale

L'obiettivo prioritario degli appunti è **agevolare, chiarire e rendere immediato ed efficace lo studio per l'esame**:

- **Le Slide sono lo scheletro**: definiscono l'indice, il perimetro degli argomenti, i grafici e le definizioni formali. Nessun argomento esterno non trattato va aggiunto.
- **La trascrizione è la sostanza**: fornisce i "perché", le spiegazioni dei passaggi sottintesi, le risposte alle domande d'aula e le analogie intuitive.
- **Tono limpido, diretto e pragmatico**:
  - **Niente gergo barocco o ampolloso**: se il docente spiega un concetto in modo chiaro, gli appunti devono conservare quella stessa limpidezza, senza travestirla da trattato accademico pesante.
  - **Nessun filler enciclopedico specialistico**: non inserire teoremi collaterali, dimostrazioni astratte o nozionismo sterile che non compaiono nelle slide e non sono stati nominati a lezione.
  - **Spiegazione cristallina delle formule**: ogni formula deve essere decodificata con cura (significato e dimensioni di ogni tensore, derivazione passaggio per riga, intuizione logico-geometrica ed esempi numerici calcolati a mano).
  - **Preservare tutte le analogie d'aula e gli esempi concreti del docente**: le metafore (es. le 32 trote, il gioco a incastro per bambini) e i calcoli reali fatti a lezione fissano i concetti nella memoria e vanno valorizzati appieno.
  - *Regola aurea: ogni riga aggiunta deve semplificare la comprensione e l'esposizione d'esame, mai aggiungere nozionismo sterile o pesantezza retorica.*

## Formule e teoremi: sempre passo passo

Per ogni formula, derivazione o teorema importante, la nota deve contenere:

1. **Cosa dice**: la formula in un riquadro (`$$\boxed{}$$` oppure `> [!abstract] Teorema`), con il significato e la forma o dimensione di ogni simbolo.
2. **Da dove viene**: la derivazione completa, un passaggio per riga, indicando la regola usata a ogni passo. Non saltare mai passaggi.
3. **Intuizione**: cosa significa in parole, con le spiegazioni e le analogie del prof integrate nel testo.
4. **Esempio numerico**: piccolo e calcolato a mano, in tabella quando aiuta. I numeri vanno verificati.
5. **Collegamenti**: wikilink alle note in cui il concetto compare già.

## Immagini

- Inserisci diagrammi e grafici delle slide con link relativi corretti rispetto alla posizione della nota (es. `../assets/...` da una sottocartella).
- Usa la slide intera se contiene annotazioni a mano, altrimenti il ritaglio della figura.
- Usa `mermaid` per i flussi.
- Controlla che ogni immagine collegata esista davvero.

## Laboratorio

- Struttura la nota in: **Teoria → Notebook cella per cella → Errori ed esperimenti**.
- Collega ogni riga di codice importante alla sua formula.
- Inserisci una tabella delle forme dei tensori.
- Interpreta i risultati confrontandoli con una baseline.
- Se il notebook è un esercizio, spiega gli errori senza correggere il file, a meno che non venga chiesto.

## Formato

- Scrivi in italiano, con un frontmatter che contiene `modulo`, `lezione`, `data`, `fonti`, `prerequisiti`, `tags`.
- Usa i callout di Obsidian solo per teoremi, avvisi, errata e riepiloghi, mai per separare la voce del prof dal resto.
- Mantieni il testo fluido, senza enfasi retorica.
- Prima di sovrascrivere una nota, salvane una copia in `_archivio/`.
- Dopo la stesura aggiorna [[00_Indice_Corso]].
