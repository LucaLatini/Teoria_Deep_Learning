---
modulo: Deep Learning - Teoria
lezione: 06 - Graph Neural Networks, GCN, applicazioni e GAT
data: [2026-10-29, 2026-10-30]
fonti:
  - "03_Fonti/Slide/06 GNN.pdf: slide stampate 1-56, pagine fisiche PDF 1-58"
  - "03_Fonti/Trascrizioni/29-10.txt: prima parte, fino alla slide 28"
  - "03_Fonti/Trascrizioni/30-10.txt: ripasso dalla slide 20; nuova parte dalla slide 30 fino alla 56"
prerequisiti: ["mlp_appunti_completi", "03_CNN", "05_Transformer"]
tags: [deep-learning, teoria, gnn, gcn, message-passing, laplaciano, fourier, chebyshev, node-classification, pooling, gat, graphcast]
---

# GNN - Graph Neural Networks

Una rete convoluzionale sfrutta la struttura dell'ingresso: un'immagine non è soltanto un elenco di valori, ma una griglia in cui certe posizioni sono vicine. Le **Graph Neural Networks**, o **GNN**, estendono questa idea a dati le cui relazioni non formano una griglia regolare. Una molecola, una rete stradale o uno scheletro hanno elementi e connessioni; la struttura di queste connessioni deve entrare nel calcolo delle rappresentazioni.

Il percorso parte dalla convoluzione già introdotta nelle [[03_CNN]], costruisce l'operatore GCN attraverso il passaggio di messaggi e ne ricostruisce la motivazione spettrale. Il risultato finale separa due operazioni: **combinare nodi collegati** e **trasformare le feature di ogni nodo**.

## 1. Dalla convoluzione alla griglia che la sostiene (slide 4-7)

### 1.1 Un filtro locale con pesi condivisi (slide 4-5)

Nella VGG, come nelle altre CNN, una successione di filtri modifica la rappresentazione dell'immagine; il classificatore usa poi le feature prodotte. Il filtro non combina pixel arbitrari: osserva una finestra locale, la sposta sistematicamente e riusa gli stessi coefficienti in tutte le posizioni. Il significato della posizione di ciascun coefficiente rimane stabile: sopra, sotto, a sinistra e a destra hanno lo stesso ruolo lungo la griglia.

Per una singola immagine a un canale e un kernel quadrato di lato $d=2k+1$, la correlazione incrociata spaziale è

$$\boxed{S(i,j)=\sum_{m=-k}^{k}\sum_{n=-k}^{k}I(i+m,j+n)K(m,n).}$$

$I(i+m,j+n)$ è il valore osservato nella finestra centrata in $(i,j)$; $K(m,n)$ è il peso all'offset $(m,n)$; $S(i,j)$ è uno scalare. Con un kernel $3\times3$, $k=1$ e vengono sommati nove prodotti. Nel caso multicanale si somma anche sui canali, secondo la convenzione del capitolo CNN.

Per esempio, per la finestra

$$I_{\mathrm{loc}}=\begin{bmatrix}1&2&3\\4&5&6\\7&8&9\end{bmatrix},\qquad K=\frac19\begin{bmatrix}1&1&1\\1&1&1\\1&1&1\end{bmatrix},$$

il risultato centrale è $(1+2+3+4+5+6+7+8+9)/9=45/9=5$. Il calcolo richiede una finestra con posizioni relative ben definite.

> [!bug] Precisazione sulla correlazione e sulla convoluzione
> La differenza matematica riguarda il ribaltamento del kernel, non la sola scelta dell'origine della finestra. La formula con $I(i+m,j+n)K(m,n)$ è una correlazione incrociata; nella convoluzione si invertono gli offset, con convenzioni coerenti. Cambiare il punto usato per indicizzare l'uscita non basta, da solo, a trasformare l'una nell'altra.

### 1.2 Reticoli regolari in due e in una dimensione (slide 6-7)

I pixel possono essere visti come punti di un reticolo discreto $\mathbb Z^2$, immerso nel piano euclideo. L'immagine associa un valore o un vettore di valori a ogni punto. La struttura delle posizioni è regolare anche se i valori dei pixel cambiano continuamente da un'immagine all'altra.

![Il reticolo regolare su cui si muove il filtro spaziale, dalla slide 6](../assets/gnn_slide_06.png)

Nella convoluzione temporale il supporto è invece un reticolo monodimensionale: le osservazioni sono ordinate lungo un asse e il filtro conserva gli stessi offset nelle diverse posizioni. Cambia il numero di assi di scorrimento, non l'idea di località e condivisione dei pesi. Audio, testo rappresentato in sequenza e serie temporali sono esempi di questo schema; il collegamento è con la convoluzione 1D già trattata nel capitolo TCN.

![Interazioni locali lungo un reticolo monodimensionale, dalla slide 7](../assets/gnn_slide_07.png)

## 2. Un'immagine come segnale su un grafo (slide 8-9)

Un **grafo** è una coppia

$$\boxed{G=(V,E),\qquad E\subseteq V\times V.}$$

$V$ è l'insieme dei nodi, o vertici; $E$ descrive gli archi, cioè le connessioni. Il grafo descrive **chi è collegato a chi**. Le feature descrivono invece **che cosa è associato a ciascun nodo**.

Per un'immagine RGB, un nodo può corrispondere a un pixel e avere come feature la terna $(R,G,B)$. Una regola di connessione a quattro vicini collega un pixel interno a quello sopra, sotto, a sinistra e a destra. Ai bordi il numero di vicini si riduce. Questo grafo è un caso particolare, molto regolare, di struttura a grafo.

![Pixel interpretati come nodi di un grafo reticolare, dalla slide 9](../assets/gnn_slide_09.png)

La connessione a quattro vicini non va confusa con tutti i pixel di un kernel $3\times3$: quest'ultimo comprende anche le diagonali e il centro. La scelta del grafo e quella del supporto del filtro sono collegate ma distinte; non basta scrivere una matrice di adiacenza a quattro vicini per riprodurre automaticamente un qualsiasi kernel CNN.

Un grafo può essere **orientato**, se il verso di una relazione conta, oppure **non orientato**, se la connessione è reciproca. Può inoltre essere binario, con archi presenti o assenti, oppure pesato. Nella derivazione spettrale successiva si considerano grafi non orientati, con adiacenza simmetrica e pesi non negativi; l'esempio più semplice è binario.

## 3. Dati irregolari e requisiti della rete (slide 10-14)

### 3.1 La località dipende dalle connessioni (slide 10-11)

Un kernel standard richiede un vicinato di dimensione e organizzazione prefissate. In un grafo generale, un nodo può avere due vicini e un altro dieci. Non esiste necessariamente una posizione universale «in alto a destra» alla quale assegnare sempre lo stesso coefficiente.

Gli esempi mostrano che la relazione utile non coincide sempre con la distanza sul foglio o nello spazio:

- In una **molecola**, gli atomi sono i nodi e i legami sono gli archi. Nell'acqua l'ossigeno è collegato a due idrogeni; nel grafo dei legami non compare un legame diretto fra i due idrogeni.
- In una **rete stradale**, due luoghi vicini in linea d'aria possono essere separati da un percorso lungo. La connessione dipende dalle strade disponibili.
- In una **rete di citazioni** o di telefonate, le relazioni dipendono da citazioni o contatti effettivi, non dalla posizione con cui si disegnano i nodi.
- In una **mesh 3D**, i vertici sono collegati lungo la superficie. Due punti qualsiasi non sono necessariamente adiacenti: per raggiungerli può servire attraversare più archi.
- Nello **scheletro umano**, i nodi rappresentano le articolazioni e gli archi le connessioni anatomiche. Due dita possono avvicinarsi nello spazio senza diventare articolazioni direttamente collegate.

![Reti stradali, citazioni, molecole e pose come esempi di dati a grafo, dalla slide 11](../assets/gnn_slide_11.png)

Dire «dominio non euclideo» non impedisce di associare ai nodi coordinate 2D o 3D. Significa che non si può ridurre la struttura delle relazioni alla sola griglia regolare delle CNN. Esiste ancora una località, ma è definita dalla **topologia del grafo**, per esempio dal numero di archi di un cammino.

### 3.2 Numerazione arbitraria dei nodi (slide 13)

Cambiare il numero assegnato a un atomo non cambia la molecola. Una rete per grafi deve rispettare questa proprietà: non deve apprendere un significato artificiale del «primo» o del «quarto» nodo. Quando si rinumerano i nodi, vanno riordinate coerentemente sia le feature sia tutte le connessioni.

La distinzione utile è fra **equivarianza** e **invarianza**. Se l'uscita contiene una rappresentazione per nodo, rinumerare l'ingresso deve rinumerare allo stesso modo le righe dell'uscita: questa è equivarianza. Se si produce un solo risultato per l'intero grafo, una successiva aggregazione indipendente dall'ordine può rendere il risultato invariante.

Non è la stessa operazione che permutare liberamente i pixel dentro un'immagine mantenendo ferma la griglia: in quel caso si cambiano i dati associati alle posizioni, e quindi il contenuto dell'immagine. Non va confusa neppure con l'invarianza alle rotazioni: se le feature contengono coordinate geometriche, una rotazione le modifica e la sola equivarianza alla rinumerazione non garantisce invarianza geometrica.

> [!bug] Precisazione slide 13 — Invarianza o equivarianza
> L'espressione generale «permutation invariance» va letta rispetto al tipo di uscita. Il livello GCN che mantiene una riga per nodo è equivarante alla rinumerazione coerente del grafo. Un'uscita globale può diventare invariante attraverso un'aggregazione appropriata.

### 3.3 Layout, vicinati e topologia variabile (slide 13-14)

Servono quindi un operatore capace di usare vicinati con cardinalità diverse, una nozione di località basata sugli archi e parametri condivisi che non dipendano dall'ordine dei nodi. Il disegno del grafo sullo schermo non costituisce un sistema di coordinate privilegiato.

![Vicinati irregolari e assenza di un ordinamento spaziale universale, dalla slide 13](../assets/gnn_slide_13.png)

La topologia può essere fissata dal dominio o variare fra esempi e situazioni. Non significa che ogni GCN debba apprendere automaticamente gli archi: nel modello qui sviluppato la matrice del grafo è fornita e i parametri appresi appartengono alla trasformazione delle feature.

## 4. GCN e passaggio di messaggi (slide 15-17)

Le GNN sono una famiglia di reti per dati a grafo. La **Graph Convolutional Network** di Kipf e Welling è una sua specifica formulazione. Come MLP e CNN, presenta ingressi, rappresentazioni intermedie, parametri, funzioni di attivazione e una loss. L'addestramento continua a usare discesa del gradiente e backpropagation.

Il principio intuitivo è il **message passing**. Ogni nodo riceve informazioni dai nodi connessi, le aggrega e costruisce una nuova rappresentazione. Questa operazione viene eseguita per tutti i nodi e può essere ripetuta in più strati.

Si possono distinguere due momenti:

1. **Aggregazione:** si combinano le rappresentazioni dei vicini. Una somma, una media o un massimo non richiedono un ordine privilegiato dei vicini. Le slide citano anche aggregazioni apprese; la formula GCN seguente usa una somma con coefficienti determinati dai gradi.
2. **Trasformazione:** l'aggregato attraversa un livello neurale e un'attivazione, producendo il nuovo embedding del nodo.

![Aggiornamenti delle rappresentazioni dei nodi tramite passaggio di messaggi, dalla slide 17](../assets/gnn_slide_17.png)

Gli aggiornamenti di uno strato usano le rappresentazioni dello **strato precedente**: non si modifica un nodo e si usa subito quel nuovo valore per aggiornare quello successivo. Usare immediatamente il valore appena aggiornato renderebbe il calcolo dipendente dall'ordine di elaborazione, invece di descrivere un aggiornamento coerente di tutto lo strato.

Gli embedding ottenuti possono essere usati da un MLP per un compito sui nodi; combinarli in un vettore globale è un'altra possibilità menzionata nel percorso introduttivo. I compiti applicativi sono sviluppati nella seconda parte del capitolo, a partire dalla slide 30.

## 5. Matrici di ingresso e ruolo dell'adiacenza (slide 18-19)

### 5.1 Feature e struttura sono due oggetti distinti (slide 18)

Con $N=|V|$ nodi e $F_0$ feature per nodo, l'ingresso è

$$\boxed{X\in\mathbb R^{N\times F_0},\qquad A\in\mathbb R^{N\times N}.}$$

Ogni riga $x_i$ di $X$ contiene le feature del nodo $i$. La matrice $A$ rappresenta gli archi: nel caso binario $A_{ij}=1$ indica una connessione e $A_{ij}=0$ la sua assenza. Nel caso non orientato $A=A^\top$; non è richiesta l'esistenza di un unico cammino fra due nodi.

![Grafo e relativa matrice di adiacenza, dalla slide 18](../assets/gnn_slide_18.png)

La figura usa un grafo orientato e una matrice non simmetrica per mostrare come registrare le connessioni. Il calcolo spettrale sviluppato nel seguito assume invece un grafo non orientato: il piccolo esempio che segue rispetta questa ipotesi.

Per il cammino a tre nodi $1-2-3$, un piccolo esempio da riusare nei calcoli è

$$A=\begin{bmatrix}0&1&0\\1&0&1\\0&1&0\end{bmatrix},\qquad
X=\begin{bmatrix}1&0\\0&2\\3&0\end{bmatrix}.$$

Ci sono tre nodi con due feature ciascuno: $X$ ha forma $3\times2$, mentre $A$ è $3\times3$. Il nodo 1 non riceve direttamente dal nodo 3 in un singolo passaggio.

Nel caso pesato, $A_{ij}$ rappresenta il peso della connessione. I pesi non sono necessariamente probabilità né devono sempre appartenere a $[0,1]$. La derivazione qui usata richiede pesi non negativi e simmetrici; non si estende invariata a qualunque matrice orientata.

### 5.2 La struttura come prior (slide 19, pagine PDF 19-20)

Una matrice di adiacenza introduce un **vincolo strutturale**: stabilisce quali informazioni possono combinarsi in un passaggio. È un inductive bias analogo, per funzione, alla località imposta dalla convoluzione sull'immagine. Può limitare relazioni non pertinenti e aiutare la generalizzazione; non garantisce automaticamente l'assenza di overfitting.

La struttura può provenire dalla conoscenza del dominio, come legami chimici e articolazioni, oppure essere costruita con euristiche, come i $k$ vicini più prossimi nello spazio delle feature o una soglia di distanza. In uno scheletro con 14 articolazioni, per esempio, l'adiacenza è $14\times14$; le feature delle articolazioni possono cambiare mentre le connessioni anatomiche restano uguali.

![Connessioni anatomiche usate come prior per una posa, dalla slide 19](../assets/gnn_slide_19.png)

## 6. Lo strato GCN in forma matriciale (slide 20-21)

### 6.1 Self-loop, gradi e normalizzazione (slide 20)

Per conservare anche il contributo del nodo stesso si aggiunge l'identità all'adiacenza:

$$\widehat A=A+I,\qquad \widehat d_i=\sum_j\widehat A_{ij},\qquad
\widehat D=\operatorname{diag}(\widehat d_1,\ldots,\widehat d_N).$$

Se $A$ non aveva self-loop, nel caso binario $\widehat d_i=d_i+1$. La matrice $\widehat D^{-1/2}$ si ottiene applicando la potenza ai soli elementi diagonali:

$$\widehat D^{-1/2}=\operatorname{diag}\left(\frac1{\sqrt{\widehat d_1}},\ldots,\frac1{\sqrt{\widehat d_N}}\right).$$

Non si eleva ogni elemento di una matrice generica alla potenza $-1/2$: la semplicità dipende dal fatto che questa matrice è diagonale. Con self-loop unitari e pesi non negativi tutti i gradi aumentati sono positivi, anche per un nodo originariamente isolato.

Definiamo l'**adiacenza normalizzata simmetricamente**

$$\boxed{\widehat S=\widehat D^{-1/2}\widehat A\widehat D^{-1/2},\qquad
\widehat S_{ij}=\frac{\widehat A_{ij}}{\sqrt{\widehat d_i\widehat d_j}}.}$$

La moltiplicazione a sinistra scala la riga $i$ di $\widehat A$ per $1/\sqrt{\widehat d_i}$; quella a destra scala la colonna $j$ per $1/\sqrt{\widehat d_j}$. Si ottiene così il denominatore con entrambi i gradi. L'operatore considera tanto il nodo che riceve quanto quello che invia il contributo.

![Slide 20 completa con le annotazioni del ripasso su message passing e normalizzazione](../assets/gnn_30_10_pdf_21_annotata.png)

> [!bug] Precisazione — Adiacenza normalizzata e Laplaciano
> $\widehat S$ non è il Laplaciano normalizzato: è l'adiacenza normalizzata del grafo con self-loop. Il corrispondente Laplaciano sarebbe $I-\widehat S$. La distinzione è essenziale per seguire segni e passaggi nella derivazione spettrale.

### 6.2 Proiezione delle feature e attivazione (slide 21)

Indichiamo con $H^{(0)}=X$ l'ingresso e con $F_\ell$ il numero di feature per nodo allo strato $\ell$. La regola coerente nelle dimensioni è

$$\boxed{H^{(\ell+1)}=\sigma\left(\widehat S H^{(\ell)}W^{(\ell)}\right).}$$

| Oggetto | Forma | Ruolo |
|---|---|---|
| $H^{(\ell)}$ | $N\times F_\ell$ | Una rappresentazione per nodo |
| $W^{(\ell)}$ | $F_\ell\times F_{\ell+1}$ | Trasformazione condivisa delle feature |
| $\widehat S$ | $N\times N$ | Aggregazione secondo la struttura del grafo |
| $H^{(\ell)}W^{(\ell)}$ | $N\times F_{\ell+1}$ | Feature proiettate |
| $H^{(\ell+1)}$ | $N\times F_{\ell+1}$ | Rappresentazioni dopo aggregazione e attivazione |

Il numero di nodi viene conservato, mentre il numero di feature può cambiare. Una ReLU agisce elemento per elemento. La matrice $W^{(\ell)}$ è condivisa fra tutti i nodi: il loro numero non moltiplica il numero dei parametri del livello. Nella formula senza bias i parametri apprendibili sono $F_\ell F_{\ell+1}$.

Per un grafo fisso, $\widehat S$ dipende dalla topologia e può essere preparata una volta; il training aggiorna $W^{(\ell)}$. Se il grafo cambia, va usata la matrice coerente con quell'ingresso. Non si possono continuare a usare coefficienti precomputati su un'altra topologia.

> [!bug] Errata slide 21 — Dimensioni dello strato precedente
> Con $W^{(\ell)}\in\mathbb R^{F_{\ell-1}\times F_\ell}$, l'ingresso della formula indicizzata con $H^{(\ell-1)}$ deve essere $N\times F_{\ell-1}$, non $N\times F_\ell$. Qui si usa uniformemente la convenzione ingresso $H^{(\ell)}$, uscita $H^{(\ell+1)}$.

### 6.3 Leggere il prodotto come passaggio di messaggi (slide 20-22, ripasso)

Se si toglie l'operatore $\widehat S$, rimane $\sigma(HW)$: ogni riga attraversa lo stesso strato fully connected e la stessa attivazione, ma non riceve informazioni dalle altre righe. È un MLP applicato separatamente ai nodi, non un meccanismo di comunicazione sul grafo. L'adiacenza è il termine che stabilisce da chi può arrivare ciascun contributo.

Per vedere la selezione senza sovrapporvi subito la normalizzazione, indichiamo con $B=HW$ la matrice delle feature già proiettate. Se la prima riga dell'adiacenza aumentata è $(1,1,0)$, il primo nodo riceve

$$(\widehat A B)_1=1\,b_1+1\,b_2+0\,b_3=b_1+b_2.$$

Con $b_1=(1,0)$, $b_2=(0,2)$ e $b_3=(3,1)$, questa riga vale $(1,2)$: il terzo vettore non compare perché manca la connessione. La moltiplicazione per l'adiacenza realizza quindi il message passing attraverso una normale somma di prodotti, non attraverso un'operazione separata esterna alla rete.

Le matrici dei gradi regolano la scala della somma. La spiegazione le presenta intuitivamente come un modo di fare una media dei contributi. Nella formula effettiva la normalizzazione è però simmetrica: ogni messaggio è pesato da $1/\sqrt{\widehat d_i\widehat d_j}$, non sempre da $1/|\mathcal N_i|$. La distinzione è già visibile nell'esempio numerico della sezione successiva, in cui una riga di $\widehat S$ non somma a uno. Nel GCN qui descritto questi coefficienti sono determinati dal grafo; sono invece appresi i pesi che trasformano le feature.

## 7. Dalla matrice ai messaggi del singolo nodo (slide 22)

### 7.1 Espansione della regola (slide 22)

Leggiamo gli embedding come vettori **riga**, coerentemente con $H$. Ponendo $\mathcal N(v)=\{u:A_{vu}\ne0\}$, nel caso binario senza self-loop preesistenti:

$$\begin{aligned}
a_v^{(\ell+1)}&=\sum_{u\in\mathcal N(v)\cup\{v\}}
\frac{h_u^{(\ell)}}{\sqrt{\widehat d_v\widehat d_u}},\\
z_v^{(\ell+1)}&=a_v^{(\ell+1)}W^{(\ell)},\\
h_v^{(\ell+1)}&=\sigma\left(z_v^{(\ell+1)}\right).
\end{aligned}$$

$a_v$ ha $F_\ell$ componenti; $z_v$ e $h_v$ ne hanno $F_{\ell+1}$. La prima riga è esattamente la riga $v$ del prodotto $\widehat SH$: gli indici dei nodi vengono sommati, l'asse delle feature resta disponibile per la proiezione.

Se il grafo è pesato, ogni termine conserva anche il numeratore $\widehat A_{vu}$. La forma con numeratore 1 non va applicata indiscriminatamente a un grafo pesato.

![Aggregazione dei vicini, proiezione e non linearità, dalla slide 22](../assets/gnn_slide_23.png)

L'associatività permette di calcolare prima $HW$ oppure prima $\widehat SH$:

$$\widehat S(HW)=(\widehat SH)W.$$

L'uguaglianza riguarda i prodotti lineari con le stesse matrici, non autorizza a spostare la ReLU fra essi. Se $W$ riduce molto le feature, proiettare prima può rendere meno costosa l'aggregazione; la scelta dipende dalle dimensioni e dalla rappresentazione sparsa del grafo.

### 7.2 Esempio numerico completo su tre nodi (slide 20-22)

Per il cammino $1-2-3$ introdotto sopra,

$$\widehat A=\begin{bmatrix}1&1&0\\1&1&1\\0&1&1\end{bmatrix},\qquad
\widehat D=\operatorname{diag}(2,3,2),$$

e quindi

$$\widehat S=\begin{bmatrix}
\frac12&\frac1{\sqrt6}&0\\
\frac1{\sqrt6}&\frac13&\frac1{\sqrt6}\\
0&\frac1{\sqrt6}&\frac12
\end{bmatrix}.$$

Scegliamo una proiezione con due feature di uscita:

$$W=\begin{bmatrix}1&-1\\2&1\end{bmatrix},\qquad
XW=\begin{bmatrix}1&-1\\4&2\\3&-3\end{bmatrix}.$$

Ogni riga viene trasformata dallo stesso $W$. Per esempio, $(0,2)W=(4,2)$. Per il nodo 1 la somma dei messaggi proiettati è

$$\begin{aligned}
z_1&=\frac12(1,-1)+\frac1{\sqrt6}(4,2)\\
&=\left(\frac12+\frac4{\sqrt6},-\frac12+\frac2{\sqrt6}\right)\\
&\simeq(2.132993,0.316497).
\end{aligned}$$

Il nodo 3 non compare direttamente. Per il nodo 2 si sommano invece tre contributi:

$$\begin{aligned}
z_2&=\frac1{\sqrt6}(1,-1)+\frac13(4,2)+\frac1{\sqrt6}(3,-3)\\
&=\left(\frac4{\sqrt6}+\frac43,-\frac4{\sqrt6}+\frac23\right)\\
&\simeq(2.966326,-0.966326).
\end{aligned}$$

Il calcolo completo e la successiva ReLU danno:

| Nodo | Prima componente di $z$ | Seconda componente di $z$ | Uscita dopo ReLU |
|---|---:|---:|---|
| 1 | 2.132993 | 0.316497 | $(2.132993,0.316497)$ |
| 2 | 2.966326 | -0.966326 | $(2.966326,0)$ |
| 3 | 3.132993 | -0.683503 | $(3.132993,0)$ |

Le righe di $\widehat S$ non sommano necessariamente a uno: qui la prima somma a $1/2+1/\sqrt6\simeq0.908248$. Questa **normalizzazione simmetrica non è una semplice media aritmetica** e non coincide con la normalizzazione per riga di una softmax.

### 7.3 Verifica della rinumerazione (slide 13 e 20-22)

Sia $P$ una matrice di permutazione: riordina le righe senza cambiarne i valori, e $P^\top P=I$. Una rinumerazione coerente produce $H'=PH$ e $\widehat S'=P\widehat SP^\top$. La nuova uscita è

$$\begin{aligned}
\sigma(\widehat S'H'W)
&=\sigma(P\widehat SP^\top PHW)\\
&=\sigma(P\widehat SHW)\\
&=P\sigma(\widehat SHW).
\end{aligned}$$

L'ultimo passaggio usa il fatto che l'attivazione è applicata componente per componente. Le rappresentazioni restano associate agli stessi nodi, soltanto in un ordine diverso: è la proprietà di equivarianza richiesta, non l'uguaglianza fra due matrici di uscita lasciate nello stesso ordine artificiale.

## 8. Profondità e campo recettivo sul grafo (slide 23)

Uno strato considera il nodo e i vicini a un arco di distanza, cioè **1-hop**. Due strati possono trasferire informazioni da nodi raggiungibili con due archi, perché ogni vicino ha già incorporato informazioni dei propri vicini. Il meccanismo è analogo all'espansione del campo recettivo nelle CNN.

Nel cammino $1-2-3$, dopo il primo strato il nodo 1 dipende da 1 e 2. Dopo il secondo può dipendere anche da 3, attraverso 2. Non viene aggiunto un arco diretto fra 1 e 3: cambia l'estensione della dipendenza, non necessariamente la topologia.

Per due livelli destinati a una classificazione dei nodi, la forma illustrata è

$$Z=\operatorname{softmax}_{\mathrm{classi}}\left(\widehat S\,\sigma(\widehat SXW^{(0)})W^{(1)}\right).$$

Se $W^{(0)}$ è $F_0\times F_1$ e $W^{(1)}$ è $F_1\times C$, $Z$ è $N\times C$. La softmax opera sulle classi di ogni nodo, non sul numero dei nodi. La distinzione fra asse delle feature e asse degli elementi è la stessa già necessaria nell'attention dei [[05_Transformer|Transformer]].

Dopo $k$ strati il campo recettivo può raggiungere nodi a distanza fino a $k$ archi. Non equivale a creare un grafo completo né garantisce che tutte le informazioni restino distinguibili. Nodi appartenenti a componenti disconnesse non possono scambiarsi messaggi attraverso un operatore che rispetta quegli archi.

La costruzione locale è chiara; resta da motivare perché questa particolare combinazione di matrici possa essere interpretata come un filtro su un grafo. Il collegamento passa dal Laplaciano, dalla trasformata di Fourier sul grafo e da una rappresentazione polinomiale del filtro.

## 9. Laplaciano normalizzato e filtri polinomiali (slide 24)

### 9.1 Operatore sul grafo e segnale sui nodi (slide 24)

Per questa derivazione torniamo inizialmente al grafo non orientato **senza** i self-loop aggiunti dalla GCN. Supponiamo $d_i=\sum_j A_{ij}>0$ e definiamo

$$\boxed{D=\operatorname{diag}(d_1,\ldots,d_N),\qquad
S=D^{-1/2}AD^{-1/2},\qquad L=I-S.}$$

$S$ è l'adiacenza normalizzata; $L$ è il **Laplaciano normalizzato**. Entrambi sono $N\times N$. Un segnale scalare sul grafo è $x\in\mathbb R^N$, con un valore $x_i$ per nodo. La derivazione si presenta prima per un segnale scalare e verrà poi estesa a più feature.

Nel caso binario, espandere la riga $i$ del prodotto dà

$$\begin{aligned}
(Lx)_i&=((I-S)x)_i\\
&=x_i-\sum_j S_{ij}x_j\\
&=x_i-\sum_{j:A_{ij}=1}\frac{x_j}{\sqrt{d_i d_j}}.
\end{aligned}$$

Il Laplaciano combina quindi già il nodo con il proprio vicinato, attraverso una differenza. Non ogni filtro coincide però con questa differenza: $Lx$ è un caso particolare, non la definizione universale di convoluzione sul grafo. L'ipotesi $d_i>0$ evita qui l'inversa di zero; nello strato finale i self-loop rendono positivi i gradi aumentati.

### 9.2 L'effetto su un autovettore (slide 24)

Un autovettore non nullo $u$ con autovalore $\lambda$ soddisfa $Lu=\lambda u$. Applicando due volte l'operatore,

$$L^2u=L(Lu)=L(\lambda u)=\lambda Lu=\lambda^2u.$$

Per ripetizione si ottiene $L^ku=\lambda^ku$. Per un polinomio $p(z)=a_0+a_1z+\cdots+a_Kz^K$,

$$\begin{aligned}
p(L)u&=(a_0I+a_1L+\cdots+a_KL^K)u\\
&=a_0u+a_1\lambda u+\cdots+a_K\lambda^Ku\\
&=p(\lambda)u.
\end{aligned}$$

Il termine costante della funzione diventa $a_0I$ nella matrice. Un polinomio del Laplaciano scala quindi ciascun autovettore con il valore assunto dal polinomio sul suo autovalore. Per esempio, con $p(z)=1-z/2$, una componente con $\lambda=0$ rimane invariata e una con $\lambda=2$ viene annullata.

## 10. Base spettrale e trasformata di Fourier sul grafo (slide 25)

### 10.1 La scomposizione e il cambio di coordinate (slide 25)

Poiché il grafo è non orientato, $L$ è reale e simmetrico. Si può scegliere una base completa di autovettori ortonormali, cioè di lunghezza uno e reciprocamente perpendicolari. Raccogliendoli per colonne,

$$\boxed{U=[u_1\ \cdots\ u_N],\quad
\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_N),\quad
L=U\Lambda U^\top,\quad U^\top U=UU^\top=I.}$$

È una decomposizione agli autovalori di una matrice simmetrica. Nel caso presente, essendo $L$ anche semidefinito positivo, è compatibile con una SVD opportunamente scelta; non si deve però estendere $U\Lambda U^\top$ a una generica matrice. Una SVD generale ha forma $U\Sigma V^\top$, con due basi eventualmente diverse.

Come nel cambio di assi usato nella PCA, un vettore può essere descritto dalle sue proiezioni sugli assi della nuova base. Qui gli assi sono gli autovettori del Laplaciano:

$$\widehat x_r=u_r^\top x,\qquad \widehat x=U^\top x,\qquad
x=U\widehat x=\sum_{r=1}^N\widehat x_ru_r.$$

Tutti i vettori sono $N\times1$. La verifica dell'inversione è immediata:

$$U\widehat x=U(U^\top x)=(UU^\top)x=Ix=x.$$

La **trasformata di Fourier sul grafo** è proprio $\widehat x=U^\top x$: riscrive il segnale in questa base spettrale. L'inversa è la moltiplicazione per $U$. Non si stanno perdendo componenti: a differenza di una PCA troncata, qui la base considerata è completa.

Gli autovalori ordinano modi di variazione sul grafo: valori bassi sono associati a componenti più regolari lungo le connessioni, quelli alti a variazioni più rapide. Si parla di frequenze del **segnale rispetto al grafo**, non di un grafo che diventa fisicamente più disordinato. Con il Laplaciano normalizzato la regolarità va intesa nella normalizzazione dei gradi; su un grafo irregolare il vettore $(\sqrt{d_1},\ldots,\sqrt{d_N})^\top$ è un autovettore a frequenza zero, non necessariamente costante sui nodi.

### 10.2 Il filtro come moltiplicazione nel dominio spettrale (slide 25)

Nel dominio di Fourier classico, la convoluzione si traduce in una moltiplicazione delle componenti trasformate. La costruzione sul grafo adotta la stessa idea: ogni frequenza riceve un moltiplicatore $g(\lambda_r)$.

$$\boxed{y=Ug(\Lambda)U^\top x,\qquad
g(\Lambda)=\operatorname{diag}\bigl(g(\lambda_1),\ldots,g(\lambda_N)\bigr).}$$

I passaggi sono tre: $U^\top$ cambia coordinate, $g(\Lambda)$ scala separatamente le componenti, $U$ ricostruisce il segnale sui nodi. Poiché la matrice del filtro è diagonale in questa base,

$$y=\sum_{r=1}^Ng(\lambda_r)\widehat x_ru_r.$$

Non occorre definire uno scorrimento geometrico del kernel su una griglia irregolare. Il filtro viene specificato attraverso il suo comportamento sulle componenti spettrali. Se $g(\lambda)=\lambda$, si ritrova $y=Lx$; un altro $g$ produce un altro filtro.

### 10.3 Esempio completo con due nodi (slide 24-25)

Due nodi collegati hanno

$$A=\begin{bmatrix}0&1\\1&0\end{bmatrix},\quad D=I,\quad
L=\begin{bmatrix}1&-1\\-1&1\end{bmatrix}.$$

Gli autovettori normalizzati sono

$$u_1=\frac1{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix},\quad \lambda_1=0,\qquad
u_2=\frac1{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix},\quad \lambda_2=2.$$

Si controlla che $Lu_1=0$ e $Lu_2=2u_2$. Il primo modo è costante sui due nodi, il secondo cambia segno: rende concreta la distinzione fra bassa e alta frequenza.

Per $x=(3,1)^\top$, le coordinate trasformate sono

$$\widehat x=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}
\begin{bmatrix}3\\1\end{bmatrix}
=\begin{bmatrix}2\sqrt2\\\sqrt2\end{bmatrix}.$$

Il filtro $g(\lambda)=1-\lambda/2$ mantiene $g(0)=1$ e annulla $g(2)=0$. Dopo la moltiplicazione spettrale rimane $(2\sqrt2,0)^\top$ e la ricostruzione restituisce $y=(2,2)^\top$. Il risultato è verificabile direttamente:

$$\left(I-\frac12L\right)x=
\begin{bmatrix}\frac12&\frac12\\\frac12&\frac12\end{bmatrix}
\begin{bmatrix}3\\1\end{bmatrix}=\begin{bmatrix}2\\2\end{bmatrix}.$$

Invece $Lx=(2,-2)^\top$: applicare soltanto il Laplaciano conserva una differenza, non produce lo stesso effetto del filtro precedente.

La formula spettrale è concettualmente utile, ma calcolare e memorizzare tutti gli autovettori di un grafo con moltissimi nodi è costoso. Il passo successivo permette di evitare questa operazione nel calcolo del filtro.

## 11. Costruire il filtro con i polinomi di Chebyshev (slide 26)

### 11.1 Base polinomiale e ricorrenza (slide 26)

Un modo per rappresentare o approssimare il filtro è usare una combinazione di polinomi. I polinomi di Chebyshev sono definiti da

$$\boxed{T_0(z)=1,\qquad T_1(z)=z,\qquad
T_k(z)=2zT_{k-1}(z)-T_{k-2}(z)\quad(k\ge2).}$$

Non serve calcolare derivate successive come in uno sviluppo di Taylor: i polinomi si costruiscono ricorsivamente. Per esempio,

$$\begin{aligned}
T_2(z)&=2zT_1(z)-T_0(z)=2z^2-1,\\
T_3(z)&=2zT_2(z)-T_1(z)=2z(2z^2-1)-z=4z^3-3z.
\end{aligned}$$

Se $z\in[-1,1]$, ciascun $T_k(z)$ rimane in $[-1,1]$. Il vincolo riguarda l'intervallo dell'argomento: non vale per qualunque $z$ reale. Per $z=1/2$, per esempio, $T_0=1$, $T_1=1/2$, $T_2=-1/2$ e $T_3=-1$.

### 11.2 Riscalare gli autovalori (slide 26)

Gli autovalori del Laplaciano normalizzato sono non negativi. Per usare la base nell'intervallo voluto, con $\lambda_{\max}>0$ si pone

$$\widetilde\lambda=\frac{2\lambda}{\lambda_{\max}}-1,\qquad
\widetilde\Lambda=\frac{2\Lambda}{\lambda_{\max}}-I.$$

La trasformazione manda $0$ in $-1$ e $\lambda_{\max}$ in $1$. Per $\lambda_{\max}=2$, gli autovalori $0,1,2$ diventano rispettivamente $-1,0,1$.

Un filtro di grado $K$ viene quindi rappresentato nella forma

$$g(\lambda)\simeq\sum_{k=0}^Kc_kT_k(\widetilde\lambda),\qquad
g(\Lambda)\simeq\sum_{k=0}^Kc_kT_k(\widetilde\Lambda).$$

Il training apprende i coefficienti $c_0,\ldots,c_K$. Si hanno **$K+1$ coefficienti**: un grado piccolo limita la famiglia dei filtri disponibili. Un filtro generale non è automaticamente un polinomio finito; la sostituzione può essere un'approssimazione. Se invece si sceglie direttamente un filtro della famiglia polinomiale, quella famiglia definisce il modello.

Con $K=1$, $\lambda_{\max}=2$ e $c_0=c_1=1/2$ si ottiene $g(\lambda)=1/2+(\lambda-1)/2=\lambda/2$. Per l'esempio a due nodi questo annulla la componente a frequenza zero e mantiene quella con autovalore 2: cambiando i coefficienti si cambia il comportamento del filtro.

## 12. Eliminare gli autovettori dal calcolo (slide 27)

### 12.1 Un'identità esatta per i polinomi (slide 27)

Definiamo il Laplaciano riscalato

$$\widetilde L=\frac{2L}{\lambda_{\max}}-I=U\widetilde\Lambda U^\top.$$

La scomposizione consente di passare dai polinomi degli autovalori ai polinomi della matrice. Il caso del quadrato mostra il meccanismo senza salti:

$$\begin{aligned}
\widetilde L^2
&=(U\widetilde\Lambda U^\top)(U\widetilde\Lambda U^\top)\\
&=U\widetilde\Lambda(U^\top U)\widetilde\Lambda U^\top\\
&=U\widetilde\Lambda I\widetilde\Lambda U^\top\\
&=U\widetilde\Lambda^2U^\top.
\end{aligned}$$

Per lo stesso motivo $\widetilde L^k=U\widetilde\Lambda^kU^\top$. Sommando le potenze con gli stessi coefficienti si ottiene l'identità matriciale

$$\boxed{UT_k(\widetilde\Lambda)U^\top=T_k(\widetilde L).}$$

Per $k=0$ la costante diventa l'identità: $UIU^\top=I$. Per $k=2$,

$$U(2\widetilde\Lambda^2-I)U^\top=2\widetilde L^2-I.$$

Queste uguaglianze sono identità fra matrici e quindi agiscono su **qualsiasi segnale** $x$ delle dimensioni corrette. La relazione più semplice $Lu=\lambda u$ descrive il singolo autovettore, ma non limita l'identità polinomiale ai soli segnali che siano autovettori.

### 12.2 Dalla formula spettrale alla ricorrenza sui segnali (slide 27)

Sostituendo lo sviluppo del filtro,

$$\begin{aligned}
y&=Ug(\Lambda)U^\top x\\
&\simeq U\left[\sum_{k=0}^Kc_kT_k(\widetilde\Lambda)\right]U^\top x\\
&=\sum_{k=0}^Kc_kUT_k(\widetilde\Lambda)U^\top x\\
&=\sum_{k=0}^Kc_kT_k(\widetilde L)x.
\end{aligned}$$

La prima sostituzione può essere approssimata; l'eliminazione di $U$ dal filtro polinomiale è esatta. Non occorre formare tutte le potenze della matrice. Si possono calcolare direttamente i vettori

$$v_0=x,\qquad v_1=\widetilde Lx,\qquad
v_k=2\widetilde Lv_{k-1}-v_{k-2},\qquad
y\simeq\sum_{k=0}^Kc_kv_k.$$

Nell'esempio a due nodi, $\lambda_{\max}=2$ dà $\widetilde L=L-I=-A$. Con $x=(3,1)^\top$:

$$v_0=\begin{bmatrix}3\\1\end{bmatrix},\qquad
v_1=\begin{bmatrix}-1\\-3\end{bmatrix},\qquad
v_2=2\widetilde Lv_1-v_0=\begin{bmatrix}3\\1\end{bmatrix}.$$

La verifica coincide con $T_2(\widetilde L)=2\widetilde L^2-I=I$, poiché qui $\widetilde L^2=I$. Con $c_0=1/2$, $c_1=-1/2$ e grado 1 si ritrova $(v_0-v_1)/2=(2,2)^\top$, senza calcolare gli autovettori durante il filtraggio.

L'interesse computazionale è proprio questo: si usano prodotti matrice-vettore, che possono sfruttare la sparsità del grafo, invece di una base spettrale completa. Per questo cammino di calcolo la diagonalizzazione serve a motivare la formula, non a essere rieseguita a ogni applicazione del livello.

## 13. Dal filtro di primo grado alla GCN finale (slide 28)

### 13.1 Due termini e un vincolo sui coefficienti (slide 28)

La formulazione di Kipf e Welling mantiene i soli termini $T_0$ e $T_1$: è un filtro di **grado uno**, con due termini, non un filtro di grado due. Per il Laplaciano normalizzato $\lambda_{\max}\le2$; usando 2 al posto del massimo effettivo,

$$\widetilde L\simeq L-I=-S.$$

Se il massimo effettivo è 2, questa riscalatura coincide con quella precedente; altrimenti è una scelta semplificata della scala. Il filtro diventa

$$\begin{aligned}
y&\simeq c_0T_0(\widetilde L)x+c_1T_1(\widetilde L)x\\
&=c_0x+c_1\widetilde Lx\\
&\simeq c_0x-c_1Sx.
\end{aligned}$$

Si impone poi il vincolo $c_0=\theta$, $c_1=-\theta$, ottenendo

$$c_0x-c_1Sx=\theta x+\theta Sx=\boxed{\theta(I+S)x.}$$

Il segno positivo nasce dai due segni meno, quello della riscalatura e quello del coefficiente. Il vincolo fra coefficienti è una **scelta del modello**, non una conseguenza obbligatoria delle proprietà di Chebyshev. L'espansione polinomiale richiamata proviene dal percorso di Defferrard e collaboratori; la semplificazione di primo ordine e la costruzione della GCN sono il passo successivo di Kipf e Welling.

### 13.2 Rinormalizzare dopo l'aggiunta dei self-loop (slide 28)

Ripetere il prodotto con $I+S$ può amplificare il segnale. Per costruire l'operatore finale si aggiungono i self-loop all'adiacenza e si ricalcolano i gradi:

$$\widehat A=A+I,\qquad
\widehat D_{ii}=d_i+1,\qquad
\widehat S=\widehat D^{-1/2}\widehat A\widehat D^{-1/2}.$$

Si sostituisce quindi $I+S$ con $\widehat S$. **Non è un'uguaglianza algebrica:** la nuova normalizzazione modifica sia il contributo del nodo stesso sia i coefficienti degli archi, perché i gradi sono cambiati.

Nel grafo a due nodi,

$$I+S=\begin{bmatrix}1&1\\1&1\end{bmatrix},\qquad
\widehat S=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix}.$$

Applicare il primo a $(3,1)^\top$ dà $(4,4)^\top$ e una seconda applicazione dà $(8,8)^\top$. Il secondo restituisce $(2,2)^\top$, che rimane uguale dopo un'altra propagazione. Il caso rende visibile tanto il cambio di operatore quanto il controllo della scala.

Per l'adiacenza simmetrica normalizzata con pesi non negativi, gli autovalori di $\widehat S$ hanno valore assoluto al più 1. Questo limita l'amplificazione dovuta alla sola propagazione; non garantisce che un'intera rete con matrici $W$ apprese e attivazioni sia sempre stabile o che non possa avere problemi di gradiente.

### 13.3 Più feature e ritorno alla formula iniziale (slide 28)

Passando da un segnale scalare a una matrice di feature, lo scalare apprendibile è sostituito da una trasformazione fra canali. Si aggiunge l'attivazione e si ritrova

$$\boxed{H^{(\ell+1)}=\sigma\left(\widehat D^{-1/2}\widehat A\widehat D^{-1/2}H^{(\ell)}W^{(\ell)}\right).}$$

La derivazione chiarisce i ruoli: $\widehat S$ combina nodi connessi, $W$ combina feature. Il primo operatore dipende dal grafo; il secondo contiene i parametri del livello. Ripetere livelli espande il campo recettivo senza richiedere un reticolo euclideo né una diagonalizzazione completa durante ogni propagazione.

La derivazione fornisce ora l'estrattore di rappresentazioni da usare nei diversi compiti su grafi. La seconda parte del capitolo introduce le teste di previsione, il pooling e l'aggregazione con attenzione.

## 14. Dal modello al compito: che cosa si vuole prevedere (slide 30)

Il message passing costruisce rappresentazioni, ma non stabilisce da solo **a quale oggetto assegnare la previsione**. Lo stesso estrattore per grafi può alimentare teste diverse: una per ciascun nodo, una per una coppia di nodi o una per l'intero grafo. Cambiano l'unità supervisionata, la forma dell'uscita e il modo in cui gli embedding vengono usati dalla loss.

![I diversi livelli a cui si può formulare un compito su grafi, dalla slide 30](../assets/gnn_30_10_pdf_31_figura.png)

| Compito | Oggetto della previsione | Esempio discusso |
|---|---|---|
| Node classification | Un'etichetta per nodo | Argomento di un documento; classe semantica di un pixel |
| Link prediction | Esistenza o tipo di relazione fra due nodi | Nuova amicizia; collegamento fra luoghi |
| Graph classification | Un'etichetta per l'intero grafo | Classe di un'immagine; azione di una persona |
| Anomaly detection | Un nodo o una configurazione con comportamento anomalo | Un computer con attività insolita nella rete |
| Community detection | Gruppi di nodi legati fra loro | Comunità nella struttura delle relazioni |

L'analogia con le immagini chiarisce la differenza. Se si considera un'immagine come grafo di pixel, classificare l'immagine significa produrre un'etichetta per tutto il segnale sul reticolo: è un compito a livello di grafo. La **segmentazione semantica** assegna invece una classe a ogni pixel, per esempio «computer» ai pixel del portatile: è l'analogo della classificazione dei nodi. Questo confronto riguarda il livello della previsione; non impone di sostituire ogni CNN con una GNN.

In una rete di computer, i nodi possono descrivere dispositivi e gli archi le comunicazioni. Un'attività molto diversa da quella attesa può suggerire un guasto o un comportamento da approfondire, anche potenzialmente malevolo; l'anomalia rilevata non identifica automaticamente una causa specifica. In una rete stradale si può invece prevedere se una coppia di luoghi ammette una certa connessione, per esempio un passaggio pedonale. Prevedere quel collegamento e calcolare il percorso migliore sono operazioni diverse.

> [!note] Precisazione sulla community detection (slide 30)
> L'espressione orale «trovare le parti più importanti» introduce intuitivamente il compito. Qui le comunità sono gruppi nella struttura delle connessioni; individuare i nodi più importanti è un'altra domanda. La lezione non sviluppa algoritmi di community detection o di anomaly detection: ne colloca soltanto il ruolo fra le applicazioni.

## 15. Classificare i nodi: uscita, applicazioni e supervisione (slide 31-35)

### 15.1 Un vettore di score per ciascun nodo (slide 31)

Se il compito riguarda ogni nodo, le trasformazioni mantengono una riga per nodo. Indicando con $F$ la dimensione finale degli embedding e con $C$ il numero di classi, la matrice finale è $H\in\mathbb R^{N\times F}$. La testa di classificazione applica la stessa trasformazione a tutte le righe:

$$\boxed{Z=HW_{\mathrm{cls}}+\mathbf 1b^\top\in\mathbb R^{N\times C},\qquad
W_{\mathrm{cls}}\in\mathbb R^{F\times C},\quad b\in\mathbb R^C.}$$

$\mathbf 1\in\mathbb R^N$ serve a ripetere il bias per tutti i nodi. La riga $z_v$ contiene i **logit**, cioè gli score prima della normalizzazione; non è ancora una distribuzione di probabilità. Per il nodo $v$:

$$\begin{aligned}
z_{v,c}&=\sum_{f=1}^Fh_{v,f}(W_{\mathrm{cls}})_{f,c}+b_c,\\
p_{v,c}&=\frac{\exp(z_{v,c})}{\sum_{k=1}^C\exp(z_{v,k})},\\
\widehat y_v&=\arg\max_c z_{v,c}=\arg\max_c p_{v,c}.
\end{aligned}$$

La prima riga è un prodotto riga per matrice; la seconda applica la softmax **sulle classi dello stesso nodo**. L'argomento del massimo non cambia, perché l'esponenziale è crescente e il denominatore è comune alle classi. Si riusano quindi il classificatore e la cross-entropy introdotti nel capitolo MLP; la GNN ha costruito le feature usando anche la struttura del grafo.

Un piccolo calcolo esplicita la testa. Con

$$h_v=(2,0),\qquad W_{\mathrm{cls}}=
\begin{bmatrix}1&0&-1\\0&1&0\end{bmatrix},\qquad b=(0,0,1)^\top,$$

si ottiene $z_v=(2,0,-1)$. Il denominatore vale $e^2+1+e^{-1}\simeq8.756935$, perciò

$$p_v\simeq(0.843795,0.114195,0.042010).$$

La previsione è la prima classe. Se è anche quella vera, il contributo alla cross-entropy è $-\log(0.843795)\simeq0.169846$. Il calcolo numerico è una ricostruzione didattica della formula, non una misura sul dataset delle slide.

> [!note] Convenzione dei vettori (slide 31)
> La slide scrive $z_v=Wh_v+b$ usando vettori colonna. Qui gli embedding della GCN sono raccolti per righe: la stessa operazione è $z_v=h_vW_{\mathrm{cls}}+b^\top$, con la matrice dei pesi trasposta rispetto alla convenzione della slide. Non cambia il classificatore.

### 15.2 Documenti collegati, profili e dataset Cora (slide 31-32)

In una rete di citazioni, ogni nodo è un documento e un arco registra una citazione. Le feature possono rappresentare titolo o abstract, mentre l'etichetta descrive l'argomento del documento. La rappresentazione di un articolo può così dipendere sia dal testo sia dai documenti collegati. Una relazione alternativa può essere la presenza di autori comuni: il grafo viene costruito in base alla relazione pertinente al problema, non soltanto al tipo di dato «documento».

Il dataset **Cora** della slide 32 usa pubblicazioni come nodi e citazioni come collegamenti. Le feature sono descritte come bag-of-words o vettori ricavati dal testo. Le sette etichette riportate sono Neural Networks, Rule Learning, Reinforcement Learning, Probabilistic Methods, Genetic Algorithms, Theory e Case-Based Reasoning. Per $N$ pubblicazioni, la testa restituisce quindi $N\times7$ score. Se la citazione viene rappresentata come relazione orientata o resa reciproca, la matrice usata dal modello deve riflettere la scelta; non si deduce la simmetria dalla sola parola «citazione».

![Rappresentazioni dei documenti visualizzate nel caso Cora, dalla slide 32](../assets/gnn_30_10_pdf_33_figura.png)

Un'altra applicazione è il profiling in una rete sociale: ogni utente è un nodo e si può prevedere una sua categoria o un attributo, per esempio un interesse. Le slide citano anche la previsione del tipo di atomo in una struttura molecolare. In tutti questi casi la GNN produce rappresentazioni per nodo; il significato dell'etichetta dipende dal dominio.

### 15.3 Siti geografici e immagini satellitari (slide 33)

Un nodo può rappresentare un **sito geografico** osservato da satellite. La sua informazione è una patch d'immagine, eventualmente con più bande; un estrattore visivo produce il vettore di feature del sito. Gli archi possono collegare siti vicini o siti con caratteristiche ambientali correlate. Il vantaggio rispetto a elaborare ogni immagine separatamente è che la descrizione di un luogo può essere aggiornata usando anche il contesto degli altri luoghi.

![Dalle immagini satellitari alle relazioni fra siti e alla classificazione, dalla slide 33](../assets/gnn_30_10_pdf_34_figura.png)

L'esempio orale motiva il problema attraverso la presenza di insetti vettori: prevedere una condizione di rischio in un sito può aiutare a scegliere dove intensificare il monitoraggio o collocare trappole. Si possono costruire relazioni diverse, basate sulla distanza geografica oppure sulla somiglianza dell'ambiente, come aree montane e aree edificate. Il segnale da classificare resta associato al sito, quindi il compito è a livello di nodo.

> [!note] Riferimento applicativo e formulazione orale (slide 33)
> La trascrizione associa il West Nile al nome reso come «Cullicoides simicola» e descrive intuitivamente un classificatore di presenza dell'insetto. L'associazione nominale non va assunta come dato biologico: l'OMS indica le zanzare del genere *Culex* come principali vettori del WNV. Inoltre, l'articolo citato nella slide, *Spotting Virus From Satellites* di Bonicelli e collaboratori (2023), ha come obiettivo la **circolazione del West Nile virus**, che non coincide con la sola presenza di un insetto. L'esempio conserva il significato didattico di una previsione per sito. Fonti di controllo della precisazione: [OMS, West Nile virus](https://www.who.int/news-room/fact-sheets/detail/west-nile-virus) e [articolo citato, DOI 10.1109/TGRS.2023.3293270](https://doi.org/10.1109/TGRS.2023.3293270).

La figura introduce anche l'idea di più relazioni fra gli stessi siti. Non si ricostruiscono qui i dettagli dell'architettura MAGAT, i protocolli sperimentali o i risultati numerici dell'articolo: non sono sviluppati nella spiegazione del caso.

### 15.4 Impostazione transduttiva e induttiva (slide 34-35)

Nel caso **transduttivo**, il grafo è disponibile durante l'addestramento, ma soltanto alcuni nodi hanno etichette note per la supervisione. L'obiettivo è prevedere le etichette dei nodi rimanenti **nello stesso grafo**. I nodi non etichettati possono comunque contribuire al message passing attraverso feature e connessioni: non devono scomparire dalla struttura solo perché manca la loro etichetta.

Se $\mathcal V_L$ è l'insieme dei nodi con etichette usate nel training, la cross-entropy supervisionata si applica soltanto a quell'insieme:

$$\boxed{\mathcal L=-\frac1{|\mathcal V_L|}\sum_{v\in\mathcal V_L}\log p_{v,y_v}.}$$

Si selezionano le righe supervisionate, si prende la probabilità della classe vera e si media il logaritmo negativo. Per esempio, con etichette note soltanto nei nodi 1 e 3 e probabilità corrette $0.8$ e $0.5$,

$$\mathcal L=-\frac{\log0.8+\log0.5}{2}\simeq0.458145.$$

Il nodo 2 può influire sugli embedding dei vicini, ma non gli si attribuisce una classe vera inventata per calcolare questa loss. La disponibilità della struttura non autorizza a usare come supervisione le etichette tenute per la valutazione.

Nel caso **induttivo**, la regola appresa deve trasferirsi a nodi o grafi non osservati durante il training. Il disegno mostra più grafi di addestramento e un nuovo grafo da etichettare. Il modello deve poter applicare i parametri condivisi alle feature e alla struttura del nuovo ingresso, senza dipendere dall'identità dei nodi del training.

![Supervisione parziale nello stesso grafo e trasferimento a nuovi grafi, dalla slide 34](../assets/gnn_30_10_pdf_35_figura.png)

> [!note] Portata dell'esempio induttivo (slide 34-35)
> La spiegazione contrappone «un grafo» a «molti grafi», come nel disegno. La distinzione generale riguarda però **che cosa resta non osservato durante il training**: anche la previsione su nuovi nodi può essere induttiva, senza richiedere per definizione molti grafi di addestramento. Occorre comunque disporre delle feature e delle relazioni che il modello usa al momento della previsione.

## 16. Link prediction: classificare una coppia di nodi (slide 36)

Nella link prediction non si legge isolatamente l'embedding di un nodo: si considerano due rappresentazioni $h_u$ e $h_v$ per stimare la loro relazione. Gli esempi sono una nuova amicizia, un'interazione futura fra utenti, il gradimento di un oggetto o un collegamento mancante. Le slide citano inoltre interazioni proteina-proteina e farmaco-bersaglio, senza sviluppare modelli specifici per questi domini.

La pipeline separa l'estrazione delle feature dal **decoder di coppia**:

$$\begin{aligned}
H&=\operatorname{GNN}(X,A),\\
s_{uv}&=\operatorname{Decoder}(h_u,h_v),\\
z_{uv}&=W_{\mathrm{edge}}s_{uv}+b_{\mathrm{edge}}\in\mathbb R^C.
\end{aligned}$$

In quest'ultima formula $s_{uv}$ è scritto come vettore colonna di dimensione $D_e$, perciò $W_{\mathrm{edge}}\in\mathbb R^{C\times D_e}$. Il decoder può restituire un vettore da classificare oppure direttamente uno score scalare. Un MLP che riceve la concatenazione $[h_u\Vert h_v]$ è un decoder apprendibile; il **prodotto scalare** è invece una scelta semplice che non introduce propri pesi aggiuntivi.

Per un solo logit binario si può scrivere

$$\boxed{r_{uv}=h_uh_v^\top=\sum_{f=1}^Fh_{u,f}h_{v,f},\qquad
p_{uv}=\operatorname{sigmoid}(r_{uv})=\frac1{1+e^{-r_{uv}}}.}$$

Il prodotto somma le interazioni componente per componente e restituisce uno scalare; la sigmoide lo porta in $(0,1)$. Gli embedding che entrano nel prodotto restano appresi dalla GNN. Per $h_u=(1,2)$ e $h_v=(2,1)$:

$$r_{uv}=1\cdot2+2\cdot1=4,\qquad p_{uv}\simeq0.982014.$$

Con etichetta $y_{uv}\in\{0,1\}$, la binary cross-entropy è

$$\mathcal L_{uv}=-y_{uv}\log p_{uv}-(1-y_{uv})\log(1-p_{uv}).$$

Se l'arco deve esserci, $y_{uv}=1$ e il contributo è circa $0.018150$; se deve essere assente, $y_{uv}=0$ e il contributo è circa $4.018150$. Lo stesso score riceve quindi un segnale di apprendimento diverso in base alla relazione vera.

> [!note] Score, classe e decoder (slide 36)
> La pipeline della slide usa $C$ score e un argomento del massimo. Per il problema binario si possono usare **due logit con softmax** oppure **un logit con sigmoide e BCE**: una softmax su un solo logit varrebbe sempre 1. Il prodotto scalare è inoltre simmetrico: scambiare $u$ e $v$ non cambia lo score. Se il tipo di relazione richiede di distinguere i due versi, questa scelta del decoder va riesaminata.

Per prevedere un collegamento sconosciuto, la coppia da valutare non deve necessariamente appartenere al vicinato usato dal message passing. Prima si costruiscono gli embedding con le relazioni disponibili; poi si interroga il decoder sulla coppia di interesse. Non si deve fornire al modello come informazione già nota proprio l'arco che si intende tenere nascosto per valutarne la previsione.

## 17. Graph classification e aggregazione dei nodi (slide 37-38)

### 17.1 Dal numero variabile di nodi a un vettore globale (slide 37)

Per assegnare una sola etichetta a un grafo occorre trasformare l'insieme degli embedding in una rappresentazione di dimensione fissa. Un grafo con 20 nodi e uno con 100 nodi possono così alimentare la stessa testa di classificazione:

$$\boxed{g=\operatorname{POOL}\{h_v:v\in V\}\in\mathbb R^F,\qquad
z_G=\operatorname{MLP}(g)\in\mathbb R^C.}$$

La previsione discreta è $\widehat y_G=\arg\max_c z_{G,c}$; il training può usare la cross-entropy con l'etichetta del grafo. Nella notazione sintetica della slide $\widehat y_G=\operatorname{MLP}(h_{\mathrm{graph}})$ è sottintesa la lettura finale degli score. Il pooling non è il classificatore: costruisce le feature globali che il classificatore userà.

La riduzione deve rispettare la numerazione arbitraria dei nodi. Riordinare le righe non deve cambiare il vettore del grafo, purché anche la struttura sia stata riordinata coerentemente. È il passaggio dall'equivarianza delle rappresentazioni per nodo all'invarianza della previsione globale, già distinto nella parte introduttiva.

### 17.2 Global average pooling (slide 38)

La riduzione più semplice è la media:

$$\boxed{g=\frac1N\sum_{v=1}^Nh_v,\qquad
g_f=\frac1N\sum_{v=1}^Nh_{v,f}.}$$

Ogni embedding ha $F$ componenti; si somma separatamente ciascuna componente e si divide per il numero di nodi. L'uscita conserva $F$ feature, perdendo l'asse dei nodi. Non occorre apprendere pesi per questa operazione. È l'analogo sul grafo del pooling globale delle mappe CNN e della media dei token del Transformer.

Con tre embedding

$$h_1=(1,0),\qquad h_2=(0,2),\qquad h_3=(3,1),$$

la somma è $(4,3)$ e $g=(4/3,1)$. Scambiare l'ordine dei tre vettori lascia invariata la somma, quindi la media. Grafo piccolo e grafo grande producono entrambi un vettore di due componenti; non conservano però necessariamente le stesse informazioni dopo la compressione.

### 17.3 Pooling con attenzione e collegamento al token speciale (slide 38)

La media attribuisce lo stesso coefficiente a tutti i nodi. Un pooling apprendibile può assegnare pesi diversi in base alle feature. Nella forma riportata dalla slide:

$$\begin{aligned}
s_v&=a_{\mathrm{pool}}^\top h_v,\\
\alpha_v&=\frac{e^{s_v}}{\sum_{u\in V}e^{s_u}},\\
\boxed{g}&=\boxed{\sum_{v\in V}\alpha_vh_v.}
\end{aligned}$$

Qui i singoli $h_v$ sono scritti come colonne, $a_{\mathrm{pool}}\in\mathbb R^F$ è apprendibile, $s_v$ e $\alpha_v$ sono scalari e $g\in\mathbb R^F$. Il prodotto scalare costruisce uno score per nodo; la softmax lo confronta con gli score di **tutti i nodi da aggregare**; la somma pesata costruisce il vettore finale. I coefficienti sono non negativi e sommano a 1.

Nell'esempio precedente, scegliendo $a_{\mathrm{pool}}=(\log2,0)^\top$, gli score sono $(\log2,0,3\log2)$ e i loro esponenziali sono $(2,1,8)$. Quindi

$$\begin{aligned}
(\alpha_1,\alpha_2,\alpha_3)&=(2/11,1/11,8/11),\\
g&=\frac2{11}(1,0)+\frac1{11}(0,2)+\frac8{11}(3,1)\\
&=(26/11,10/11)\simeq(2.363636,0.909091).
\end{aligned}$$

Il risultato privilegia il terzo nodo per effetto degli score, non perché si trovi artificialmente in terza posizione. Il vettore dei pesi è un esempio scelto per rendere il calcolo verificabile; durante il training viene appreso.

Una seconda realizzazione discussa usa una **query speciale**, come il token [CLS] del Transformer. Si può aggiungere un nodo dedicato alla raccolta globale, collegarlo ai nodi del grafo e leggere la sua rappresentazione finale. Per un'operazione di attention di lettura, con una query riga $q\in\mathbb R^{1\times d_k}$, chiavi $K\in\mathbb R^{N\times d_k}$ e valori $V\in\mathbb R^{N\times F_o}$:

$$\begin{aligned}
qK^\top&\in\mathbb R^{1\times N},\\
\beta&=\operatorname{softmax}_{\mathrm{nodi}}(qK^\top/\sqrt{d_k})\in\mathbb R^{1\times N},\\
g&=\beta V\in\mathbb R^{1\times F_o}.
\end{aligned}$$

**Una query produce una riga di uscita**: è il motivo per cui questa costruzione può ridurre il grafo a un vettore. Se $N=3$, $F_o=2$, $\beta=(2/11,1/11,8/11)$ e le righe di $V$ sono i tre embedding dell'esempio, il prodotto restituisce lo stesso $(26/11,10/11)$. La formula con $a_{\mathrm{pool}}$ e la lettura con query speciale sono due possibili costruzioni del pooling con attenzione; non sono per definizione la stessa parametrizzazione.

La scelta di usare un nodo nuovo o di leggere un'articolazione già esistente, per esempio la testa nello scheletro, è una scelta architetturale. Quel nodo deve avere accesso alle informazioni che si vogliono raccogliere; un'articolazione qualsiasi non diventa automaticamente un riassunto globale senza collegamenti e addestramento adeguati.

### 17.4 Pooling gerarchico (slide 38)

Un unico pooling globale comprime tutto il grafo in un passaggio. Il **pooling gerarchico** divide invece i nodi in gruppi e costruisce un nodo riassuntivo per ciascun gruppo. Si ottiene un grafo più piccolo, sul quale possono seguire altre trasformazioni e altre aggregazioni. È l'analogo dell'aumento graduale della scala nelle CNN, adattato a una struttura senza finestre rettangolari universali.

![Aggregazioni successive di gruppi di nodi, dalla slide 38](../assets/gnn_30_10_pdf_39_figura.png)

Per esempio, si può partizionare il grafo in cinque cluster e fare la media separatamente in ciascuno. Per un cluster $\mathcal C_r$, il vettore risultante è

$$g_r=\frac1{|\mathcal C_r|}\sum_{v\in\mathcal C_r}h_v.$$

Con i tre nodi dell'esempio, i gruppi $\{1,2\}$ e $\{3\}$ darebbero rispettivamente $(1/2,1)$ e $(3,1)$. Sono ora due rappresentazioni di gruppi; le relazioni del grafo ridotto devono descrivere come tali gruppi sono collegati. Non è la stessa operazione che applicare la media sull'intero insieme fin dall'inizio. La lezione introduce la costruzione per clustering e aggregazioni successive, senza sviluppare uno specifico algoritmo di clustering o di pooling appreso.

Un cenno orale collega inoltre l'analisi dei flussi su grafi alla rilevazione di comportamenti anomali negli LLM, anche associati ad allucinazioni. Non sono indicati uno studio o un protocollo sufficienti a ricostruire quel metodo: resta un esempio di possibile applicazione, senza aggiungere un'architettura non descritta nelle fonti.

## 18. Action recognition con grafi spazio-temporali (slide 39)

### 18.1 Dal video alle coordinate delle articolazioni (slide 39)

Un video contiene molti pixel, ma per riconoscere alcune azioni può essere utile rappresentare direttamente **come si muove il corpo**. Un passaggio di pose estimation estrae, per ogni fotogramma, le posizioni delle articolazioni. La sequenza diventa così una successione di scheletri, più compatta del video originale.

Le posizioni possono provenire da un sensore dedicato, come nell'esempio Kinect, oppure da una rete che prende l'immagine e stima le coordinate dei giunti. Sono due modi di ottenere l'ingresso per il modello successivo. Una stima di posa resta una misura o una previsione con possibili errori: non va confusa con la conoscenza esatta della posizione reale.

Con $T$ istanti, $J$ articolazioni e coordinate in $d=2$ o $d=3$ dimensioni, un modo di raccogliere il segnale è

$$X\in\mathbb R^{T\times J\times d},\qquad x_{t,j}\in\mathbb R^d.$$

Ogni $x_{t,j}$ descrive la posizione dell'articolazione $j$ al tempo $t$. Nell'esempio orale di uno scheletro con 24 articolazioni, quattro fotogrammi e coordinate 3D, il tensore ha forma $4\times24\times3$: contiene 288 coordinate. Il numero di articolazioni dipende dalla rappresentazione della posa, non è una costante obbligatoria di tutte le ST-GCN.

### 18.2 Archi spaziali e archi temporali (slide 39)

Si possono vedere le coppie $(t,j)$ come nodi di un unico grafo spazio-temporale. I nodi sono $TJ$; la matrice di feature può essere scritta anche come $X_{\mathrm{flat}}\in\mathbb R^{(TJ)\times d}$, senza cambiare il segnale rappresentato. Le relazioni hanno due significati:

- Gli **archi spaziali** collegano articolazioni connesse nello stesso istante, per esempio polso e gomito. Nella figura sono blu.
- Gli **archi temporali** collegano la stessa articolazione in istanti successivi, per esempio il polso al tempo $t$ e il polso al tempo $t+1$. Nella figura sono verdi.

![Slide 39 completa con annotazione sulle connessioni temporali verdi](../assets/gnn_30_10_pdf_40_annotata.png)

Per esempio, se il polso passa da $(1,0,0)$ a $(2,0,0)$, la variazione è $(1,0,0)$: l'arco temporale mette in relazione le due osservazioni della stessa articolazione. Nello stesso fotogramma, l'arco verso il gomito permette invece di descrivere la configurazione del braccio. I due collegamenti portano informazioni diverse, entrambe utili per distinguere un'azione.

Le **Spatio-Temporal Graph Convolutional Networks**, o **ST-GCN**, sfruttano questa struttura per aggiornare le rappresentazioni. Si può fornire una sola struttura complessiva oppure mantenere distinta la natura spaziale e temporale degli archi. La distinzione può aiutare il modello a elaborare separatamente relazioni con ruoli diversi; non implica una garanzia universale di prestazioni migliori per ogni dataset.

### 18.3 Un'azione come etichetta dell'intera sequenza (slide 39)

Il risultato richiesto è la classe dell'azione, per esempio una categoria di movimento, non un'etichetta indipendente per ciascuna articolazione. Dopo gli strati spazio-temporali si aggregano le rappresentazioni dei nodi e si usa una testa di classificazione del grafo. È quindi un caso di **graph classification**, anche se la struttura comprende più fotogrammi.

La pipeline è: video, stima delle pose, coordinate e connessioni spazio-temporali, strati per grafi, pooling e classificatore. Dopo aver estratto la posa, questo modello lavora sulla struttura degli scheletri; non deve necessariamente continuare a leggere tutti i pixel del video.

La stessa rappresentazione si può interpretare come un insieme di **serie temporali**: una per articolazione, con due o tre coordinate per istante. Il grafo aggiunge però le relazioni fra quelle serie. Non si osserva soltanto la traiettoria del singolo polso; si possono apprendere pattern che dipendono dalla sua interazione con gomito, spalla e altre articolazioni. È il collegamento con RNN e TCN: l'evoluzione temporale rimane centrale, ma qui è accompagnata da un prior esplicito sulle relazioni del corpo.

## 19. Limiti dell'aggregazione GCN e motivazione dell'attenzione (slide 41-42)

Nella GCN già costruita, la trasformazione $W$ è appresa, mentre i coefficienti di aggregazione dipendono dall'adiacenza e dai gradi. Due messaggi con ruoli diversi non ricevono un peso scelto confrontando il loro contenuto: il filtro strutturale è fissato dal grafo. «Tutti i vicini contribuiscono ugualmente», nella formulazione sintetica della slide 41, va quindi letto come assenza di attenzione al contenuto; con la normalizzazione simmetrica i coefficienti possono già essere numericamente diversi a causa dei gradi.

Un secondo limite è l'**over-smoothing**. Ripetere molte aggregazioni può rendere le rappresentazioni dei nodi sempre meno distinguibili. Aumentare la profondità estende il campo recettivo, ma non assicura di mantenere differenze utili alla classificazione: la rete può mescolare eccessivamente i segnali.

Un esempio minimo rende visibile il fenomeno. Per due nodi connessi con self-loop, senza una trasformazione ulteriore e con attivazione identità,

$$\widehat S=\frac12\begin{bmatrix}1&1\\1&1\end{bmatrix},\qquad
H=\begin{bmatrix}0\\2\end{bmatrix},\qquad
\widehat SH=\begin{bmatrix}1\\1\end{bmatrix}.$$

Le due rappresentazioni diventano uguali già in un passaggio e successive applicazioni di questo operatore le mantengono uguali. È un caso illustrativo di mescolamento, non la dimostrazione che ogni GCN debba collassare dopo uno specifico numero di strati. Nelle reti complete intervengono anche pesi, attivazioni e struttura del grafo.

Infine la GCN richiede una struttura disponibile o costruita: non bastano le sole feature se non si è deciso quali nodi possano comunicare. Le **Graph Attention Networks**, o **GAT**, introducono coefficienti di aggregazione appresi a partire dalle rappresentazioni dei nodi. Si vuole quindi imparare anche **quanto contano i diversi messaggi**, oltre a come trasformare le feature.

> [!note] Attenzione e struttura disponibile (slide 41-44)
> Il richiamo orale a «imparare l'adiacenza» descrive i coefficienti di una **matrice di interazione pesata**, non implica necessariamente apprendere la presenza discreta di tutti gli archi. Se viene fornita una maschera, il modello pesa soltanto i collegamenti ammessi. Senza una maschera si possono considerare tutte le coppie, assumendo un insieme completo di interazioni candidate. L'attenzione, da sola, non garantisce inoltre di eliminare l'over-smoothing in qualsiasi rete profonda.

![Più pesi di attenzione per i messaggi diretti allo stesso nodo, dalla slide 42](../assets/gnn_30_10_pdf_43_figura.png)

## 20. Un livello GAT: proiezione, score, maschera e aggregazione (slide 43-45)

### 20.1 Prima trasformare le feature (slide 43, pagine PDF 44-45)

Per seguire la notazione delle slide GAT, in questa derivazione il singolo embedding è un **vettore colonna**: $h_i\in\mathbb R^F$. La trasformazione condivisa è

$$\boxed{u_i=Wh_i\in\mathbb R^{F'},\qquad W\in\mathbb R^{F'\times F}.}$$

La slide indica questo vettore con $h_i^W$; qui $u_i$ evita di confondere l'apice con una potenza. Ogni componente di uscita è una combinazione delle $F$ feature di ingresso. Lo stesso $W$ è usato per tutti i nodi, come per la proiezione della GCN. Per esempio, con $F=F'=2$ e $W=I$, un embedding $(1,0)^\top$ resta $(1,0)^\top$; con $W=\operatorname{diag}(2,1)$ diventerebbe $(2,0)^\top$.

Questo cambio di convenzione è locale e dichiarato: se si raccolgono nuovamente gli embedding per righe in $H\in\mathbb R^{N\times F}$, la proiezione complessiva è $U=HW^\top\in\mathbb R^{N\times F'}$, non $HW$ con la matrice delle slide lasciata nelle stesse dimensioni.

### 20.2 Dalla coppia di vettori a uno score scalare (slide 43)

Per misurare il contributo del nodo $j$ all'aggiornamento di $i$, si concatenano le due rappresentazioni proiettate:

$$[u_i\Vert u_j]\in\mathbb R^{2F'},\qquad a\in\mathbb R^{2F'}.$$

Il vettore $a$ è apprendibile. La concatenazione unisce le componenti senza sommarle; il prodotto scalare successivo le combina in **un solo numero**:

$$\boxed{e_{ij}=\operatorname{LeakyReLU}\left(a^\top[u_i\Vert u_j]\right).}$$

Scomponendo $a=[a_L\Vert a_R]$, con $a_L,a_R\in\mathbb R^{F'}$,

$$\begin{aligned}
a^\top[u_i\Vert u_j]&=\sum_{r=1}^{F'}(a_L)_r(u_i)_r+
\sum_{r=1}^{F'}(a_R)_r(u_j)_r\\
&=a_L^\top u_i+a_R^\top u_j.
\end{aligned}$$

La Leaky ReLU conserva la natura scalare dello score. Con pendenza negativa $\eta\in(0,1)$, vale $\operatorname{LeakyReLU}(x)=x$ per $x\ge0$ e $\eta x$ per $x<0$. Diversamente dalla ReLU standard, non annulla tutto il semiasse negativo. Il richiamo è alle funzioni di attivazione del capitolo CNN.

Per un esempio da usare anche nei passaggi successivi, fissiamo

$$h_1=\begin{bmatrix}1\\0\end{bmatrix},\quad
h_2=\begin{bmatrix}0\\2\end{bmatrix},\quad
h_3=\begin{bmatrix}-1\\1\end{bmatrix},\quad
W=I,\quad a=\begin{bmatrix}1\\-1\\1\\1\end{bmatrix},\quad\eta=0.2.$$

Per la coppia ordinata $(1,2)$,

$$a^\top[u_1\Vert u_2]=1\cdot1-1\cdot0+1\cdot0+1\cdot2=3,$$

quindi $e_{12}=3$. Per $(2,1)$ il valore prima dell'attivazione è $-1$ e $e_{21}=-0.2$. Lo score può dunque essere diverso nei due versi anche con un grafo non orientato. Questi valori sono scelte didattiche per il calcolo, non parametri appresi nel caso Cora.

Lo score $e_{ij}$ non è ancora il coefficiente del messaggio, né una probabilità di esistenza dell'arco. È un'affinità da normalizzare insieme alle altre della stessa riga. I pesi $W$ e $a$ sono parametri del modello; gli score cambiano quando cambiano gli embedding in ingresso.

### 20.3 Quali coppie considerare e che cosa significa la maschera (slide 44)

Nella versione locale si definisce $\mathcal N_i$ come l'insieme dei nodi da cui $i$ può ricevere un messaggio, includendo il nodo stesso se si usano self-loop. Per un grafo orientato fissiamo la convenzione: la riga $i$, colonna $j$ descrive il contributo **$j\to i$**. È questa convenzione, insieme alla maschera, a stabilire il verso del flusso.

$$\widetilde e_{ij}=\begin{cases}
e_{ij},&j\in\mathcal N_i,\\
-\infty,&j\notin\mathcal N_i.
\end{cases}$$

Il valore $-\infty$ non è il peso di un messaggio: è un valore messo **prima della softmax** per rendere nullo quel contributo dopo la normalizzazione. Mettere semplicemente 0 prima della softmax non avrebbe lo stesso effetto, perché $e^0=1$.

Per il cammino $1-2-3$ con self-loop, il supporto ammesso è

$$M=\begin{bmatrix}1&1&0\\1&1&1\\0&1&1\end{bmatrix}.$$

La coppia $(1,3)$ è esclusa, ma $(1,1)$ e $(1,2)$ sono ammesse. Con i dati numerici precedenti, la matrice degli score mascherati è

$$\widetilde E=\begin{bmatrix}
2&3&-\infty\\
-0.2&0&-0.4\\
-\infty&0&-0.4
\end{bmatrix}.$$

Per esempio, nella seconda riga la concatenazione con il nodo 3 dà $-2$ prima della Leaky ReLU, quindi $-0.4$. Tutte le posizioni non ammesse rimangono escluse, indipendentemente dal loro eventuale score non mascherato.

La domanda annotata sulle coppie va letta in termini di **coppie ordinate**: $e_{ij}$ ed $e_{ji}$ possono differire e si considerano anche i self-loop se ammessi. Nella versione completa ci sono $N^2$ posizioni candidate; in quella mascherata bastano le posizioni del supporto ammesso. Non è necessario calcolare tutte le coppie non collegate per poi scartarle. La possibilità di usare una maschera permette inoltre di lavorare con relazioni orientate senza imporre la simmetria richiesta dalla precedente derivazione spettrale della GCN.

### 20.4 Softmax sui vicini: perché il contributo escluso diventa zero (slide 45)

Per ogni destinatario $i$ si normalizzano gli score sui nodi sorgente:

$$\boxed{\alpha_{ij}=\frac{\exp(\widetilde e_{ij})}{\sum_{k=1}^N\exp(\widetilde e_{ik})}
=\frac{\exp(e_{ij})}{\sum_{k\in\mathcal N_i}\exp(e_{ik})}
\quad(j\in\mathcal N_i).}$$

Per $j\notin\mathcal N_i$, il numeratore è $\exp(-\infty)=0$, quindi $\alpha_{ij}=0$. Per una riga con almeno una posizione valida, la somma dei coefficienti è

$$\sum_j\alpha_{ij}=\frac{\sum_j\exp(\widetilde e_{ij})}{\sum_k\exp(\widetilde e_{ik})}=1.$$

Nell'esempio, la prima riga diventa

$$\begin{aligned}
\alpha_{11}&=\frac{e^2}{e^2+e^3}=\frac1{1+e}\simeq0.268941,\\
\alpha_{12}&=\frac{e^3}{e^2+e^3}=\frac e{1+e}\simeq0.731059,\\
\alpha_{13}&=0.
\end{aligned}$$

Il coefficiente maggiore corrisponde allo score maggiore; la somma della riga è uno; la connessione mascherata non contribuisce. Sono i tre effetti della normalizzazione discussi nella spiegazione. Il denominatore non somma tutte le righe della matrice: confronta i candidati della **stessa riga**.

> [!note] Righe senza vicini ammessi (slide 44-45)
> La proprietà precedente richiede un denominatore non nullo. Se una riga fosse interamente mascherata, la softmax non definirebbe una distribuzione valida. Includere un self-loop è un modo per garantire almeno un candidato per nodo; un'implementazione deve comunque gestire esplicitamente il proprio supporto.

### 20.5 Somma pesata e nuova rappresentazione del nodo (slide 45)

Dopo la normalizzazione, i coefficienti si applicano alle feature proiettate:

$$\boxed{h'_i=\sigma\left(\sum_{j\in\mathcal N_i}\alpha_{ij}u_j\right)
=\sigma\left(\sum_{j\in\mathcal N_i}\alpha_{ij}Wh_j\right).}$$

Ogni $\alpha_{ij}$ è uno scalare e ogni $u_j$ ha $F'$ componenti. Il prodotto scala tutte le componenti del messaggio, la somma restituisce un vettore in $\mathbb R^{F'}$ e l'attivazione opera componente per componente. È sempre message passing: cambia il criterio con cui si pesano le informazioni disponibili.

Se $U$ raccoglie i vettori proiettati per righe e $\mathbf A_{\mathrm{att}}=[\alpha_{ij}]$, l'intero strato si legge come

$$\boxed{H'=\sigma(\mathbf A_{\mathrm{att}} U)=\sigma(\mathbf A_{\mathrm{att}} H W^\top).}$$

Le dimensioni sono $(N\times N)(N\times F')=N\times F'$. Il parallelo con $\widehat S H W$ della GCN è diretto: $\mathbf A_{\mathrm{att}}$ svolge il ruolo di operatore di aggregazione, ma viene calcolata dalle feature e dai parametri dell'attenzione.

![Slide 45 completa con schema del calcolo e annotazione sulle coppie di nodi](../assets/gnn_30_10_pdf_47_annotata.png)

Nell'esempio precedente, con ReLU come attivazione finale, si ottengono:

| Nodo | Coefficienti verso i nodi 1, 2, 3 | Somma prima della ReLU | Nuova rappresentazione |
|---|---|---|---|
| 1 | $(0.268941,0.731059,0)$ | $(0.268941,1.462117)$ | $(0.268941,1.462117)$ |
| 2 | $(0.328933,0.401760,0.269307)$ | $(0.059625,1.072827)$ | $(0.059625,1.072827)$ |
| 3 | $(0,0.598688,0.401312)$ | $(-0.401312,1.598688)$ | $(0,1.598688)$ |

Per il primo nodo, il calcolo è $0.268941(1,0)+0.731059(0,2)$; per il secondo è la combinazione di tutti e tre i vettori; per il terzo è $0.598688(0,2)+0.401312(-1,1)$. La ReLU modifica soltanto la prima componente negativa dell'ultima riga. Gli arrotondamenti della tabella sono effettuati dopo i calcoli.

## 21. Più teste, confronto con il Transformer e limiti della GAT (slide 46-50)

### 21.1 Multi-head graph attention e residui (slide 46)

Una sola testa usa una proiezione e un vettore di affinità. Più teste apprendono in parallelo trasformazioni e coefficienti differenti sugli stessi nodi. Con $K$ teste e $F'$ feature prodotte da ciascuna:

$$\boxed{h'_i=\mathop{\Vert}_{k=1}^K
\sigma\left(\sum_{j\in\mathcal N_i}\alpha_{ij}^{(k)}W^{(k)}h_j\right)
\in\mathbb R^{KF'}.}$$

Si esegue una somma pesata per ogni testa e si concatenano i risultati **lungo le feature dello stesso nodo**. Tre teste da quattro feature producono 12 feature per nodo, quindi $N\times12$ per tutto il grafo: non producono tre volte il numero di nodi. Per due teste con uscite $(1,2)$ e $(3,4)$ su uno stesso nodo, la concatenazione è $(1,2,3,4)$, non $(4,6)$: quest'ultimo sarebbe un risultato di somma, che è un'altra operazione.

Le GAT possono essere combinate con connessioni residue. Per una somma residua devono coincidere le forme dei due rami, eventualmente adattando le dimensioni con una proiezione. Vale la stessa condizione già incontrata in ResNet e nei Transformer; aggiungere il collegamento non significa saltare automaticamente il calcolo del ramo trasformato.

### 21.2 Che cosa condivide con la self-attention e che cosa cambia (slide 47)

GAT e self-attention costruiscono entrambe coefficienti per combinare rappresentazioni. Nel GAT descritto, lo stesso vettore proiettato $Wh_i$ partecipa al calcolo delle affinità e, quando il nodo invia un messaggio, al contenuto aggregato. Nel Transformer studiato, query, key e value sono prodotti da tre proiezioni distinte.

| Aspetto | GAT descritta | Self-attention del Transformer studiato |
|---|---|---|
| Score della coppia | Leaky ReLU di una proiezione della concatenazione | Prodotto query-key diviso per $\sqrt{d_k}$ |
| Trasformazioni delle feature | Un $W$ condiviso nei ruoli precedenti e vettore $a$ | Proiezioni distinte $W^Q,W^K,W^V$ |
| Supporto delle interazioni | Vicini ammessi dalla maschera, oppure tutte le coppie | Tutte le coppie ammesse dall'eventuale maschera |
| Struttura fornita al modello | Relazioni del grafo | Posizioni dei token e maschere del problema |
| Aggregazione | Somma pesata dei vettori proiettati | Somma pesata dei value |

Il richiamo orale alla «condivisione fra query, key e value» aiuta a confrontare i ruoli, ma la GAT **non è soltanto una scaled dot-product attention con tre matrici rese uguali**: cambia anche la funzione che costruisce lo score. Nella singola testa GAT, senza bias, i parametri di proiezione e affinità sono $F'F+2F'$. Per $F=4$ e $F'=3$ sono $12+6=18$, condivisi fra le coppie. Il confronto con i parametri di una testa Transformer dipende dalle dimensioni delle sue proiezioni e dall'eventuale proiezione finale; non basta contare le matrici ignorandone le forme.

La tabella delle slide contrappone il caso usuale di GAT locale e sparsa alla self-attention completa su una sequenza. È un confronto fra le configurazioni illustrate, non un divieto per i Transformer di usare maschere. Se si elimina la maschera della GAT, non si conserva più il vincolo di località imposto dagli archi; possono comunque restare informazioni geometriche nelle feature dei nodi. L'adiacenza, d'altra parte, non sostituisce automaticamente qualunque informazione sulle coordinate o sull'orientamento.

### 21.3 Pesi osservabili, interpretabilità e GATv2 (slide 48-50)

I coefficienti di attenzione sono quantità che si possono osservare. La slide 48 mostra le rappresentazioni di un livello GAT su Cora: i colori indicano le classi dei nodi e lo spessore dei collegamenti riassume coefficienti di attenzione attraverso le teste e i due versi. È una visualizzazione utile per esaminare il comportamento del modello, non una prova che quei pesi costituiscano da soli una spiegazione causale della decisione.

![Visualizzazione di rappresentazioni e pesi di attenzione su Cora, dalla slide 48](../assets/gnn_30_10_pdf_50_figura.png)

> [!info] Dettagli presenti nelle slide, soltanto richiamati nella spiegazione (slide 48-50)
> La visualizzazione usa t-SNE, ma il suo algoritmo non viene sviluppato. GATv2 è menzionata come evoluzione del modello; le slide 49-50 ne esplicitano il limite e la modifica delle operazioni, senza una derivazione orale. Nella GAT standard lo score ha forma $\operatorname{LeakyReLU}(a_L^\top Wh_i+a_R^\top Wh_j)$: a parità di candidati, il primo termine è comune a tutti i vicini della stessa query e la Leaky ReLU è monotona. L'ordinamento dei vicini non può quindi essere scelto arbitrariamente in funzione della query. Questo è il significato dell'attenzione «statica» indicata nelle slide, non il fatto che tutti i coefficienti numerici restino sempre costanti.

La formula di GATv2 riportata nella slide 50 cambia l'ordine fra proiezione, non linearità e riduzione a scalare:

$$e^{\mathrm{GATv2}}_{ij}=\widetilde a^\top
\operatorname{LeakyReLU}\left(\widetilde W[h_i\Vert h_j]\right).$$

Qui $[h_i\Vert h_j]\in\mathbb R^{2F}$, $\widetilde W\in\mathbb R^{D_a\times2F}$ e $\widetilde a\in\mathbb R^{D_a}$: la non linearità agisce su un vettore prima della combinazione finale. La formula resta materiale presente nelle slide; non si aggiungono una dimostrazione di approssimazione universale, confronti sperimentali o un'implementazione di GATv2 non svolti nella lezione. Il limite chiarisce anche perché la descrizione intuitiva di GAT come attenzione adattabile non debba essere letta come una capacità illimitata di modificare ogni ordinamento dei vicini.

## 22. GraphCast: geometria della Terra e previsione meteorologica (slide 52-56)

### 22.1 Prevedere l'evoluzione dello stato atmosferico (slide 52-53)

Nella previsione meteorologica tradizionale, un simulatore fisico evolve uno stato dell'atmosfera attraverso modelli dei processi ambientali. Il calcolo può essere costoso. L'esempio **GraphCast**, presentato nelle slide con riferimento al lavoro del 2023, mostra una diversa via: apprendere dai dati una regola di previsione che utilizza una rappresentazione a grafo della geometria terrestre.

I nodi descrivono posizioni o regioni su una discretizzazione della superficie e le feature contengono variabili meteorologiche, come vento, temperatura e pressione. Gli archi descrivono relazioni geometriche fra le posizioni. Il target è il valore futuro di variabili associate alle posizioni: è una previsione a livello di nodo, ma di quantità numeriche, non una classificazione di tutta la Terra in una sola categoria.

La figura mostra stati meteorologici precedenti e correnti che alimentano la previsione dello stato successivo. In forma schematica:

$$\widehat X_{t+\Delta}=f_\theta(X_{t-\Delta},X_t;G_{\mathrm{geom}}).$$

$X_t$ raccoglie le variabili allo stesso istante, $\Delta$ indica il passo temporale e $G_{\mathrm{geom}}$ la struttura geometrica usata nel modello. Se il segnale sulla griglia di ingresso ha $N_g$ posizioni e $F_m$ variabili, $X_t$ e il target sulla stessa griglia hanno forma $N_g\times F_m$. La formula descrive la dipendenza temporale del problema, non tutte le operazioni interne di trasferimento fra griglia e mesh.

![Dagli stati osservati alla previsione successiva e alla sua ripetizione nel tempo, dalla slide 53](../assets/gnn_30_10_pdf_55_figura.png)

Per esempio, con tre posizioni e due variabili per posizione, lo stato da prevedere contiene sei valori. Si apprendono le trasformazioni che collegano gli stati osservati a quelli futuri; il target del training è lo stato successivo disponibile nei dati. L'esempio di forma non aggiunge uno specifico metodo di loss o di calibrazione non sviluppato nella spiegazione.

Il processo può essere ripetuto per ottenere una traiettoria futura, come nel rollout della figura. I parametri restano quelli del modello addestrato; si aggiornano gli stati usati nella previsione. Le slide indicano un addestramento su decenni di dati meteorologici e riportano un confronto competitivo con HRES per previsioni a dieci giorni, insieme a un'inferenza molto più rapida. Sono risultati del confronto presentato sul modello del 2023, non una garanzia universale per qualsiasi variabile, situazione o sistema meteorologico.

> [!note] Simulazione, apprendimento e accuratezza (slide 52-53)
> La spiegazione orale descrive l'obiettivo come «imparare la simulazione» e qualifica alcune previsioni fisiche come «exact or very accurate». Il senso utile è apprendere l'evoluzione futura senza ripetere tutte le operazioni di un simulatore a ogni previsione. Le slide parlano di addestramento su dati meteorologici: non si deve dedurre che il modello copi esclusivamente l'uscita di un singolo simulatore, né che le previsioni fisiche o apprese siano esatte. La maggiore velocità riguarda la previsione del modello già addestrato; il training ha un proprio costo.

### 22.2 Mesh a più risoluzioni e passaggio di messaggi (slide 54)

La Terra viene rappresentata mediante una **mesh**, cioè una discretizzazione di punti e connessioni sulla superficie sferica. Le slide mostrano strutture icosaedriche a più risoluzioni. Una risoluzione maggiore descrive dettagli più fini, ma comporta più nodi e più calcolo. Una struttura più grossolana copre regioni più ampie con meno elementi.

![Trasferimento fra griglia e mesh e connessioni a più scale, dalla slide 54](../assets/gnn_30_10_pdf_56_figura.png)

L'architettura illustrata distingue un trasferimento delle informazioni verso la mesh, un processore basato sul message passing e una ricostruzione verso l'uscita. Le connessioni su più scale permettono di combinare contesto esteso e variazioni locali, invece di scegliere soltanto una discretizzazione fine o soltanto una grossolana. La rappresentazione multirisoluzione viene usata nello stesso modello; il disegno non impone di interpretarla come una serie di modelli indipendenti addestrati separatamente.

L'analogia con le gerarchie già studiate riguarda le scale dell'informazione: un fenomeno locale può essere influenzato da configurazioni più ampie. Nel caso meteorologico la geometria delle relazioni è parte del problema, non un ordinamento arbitrario imposto a un elenco di valori. Il capitolo non sviluppa tutte le equazioni interne di GraphCast o i dettagli numerici del suo simulatore di confronto, perché il caso è presentato come applicazione intuitiva delle GNN.

### 22.3 Il ruolo del prior geometrico nell'applicazione (slide 55-56)

Il motivo per scegliere una rete per grafi emerge qui in modo concreto: quando il target dipende da **come le unità sono connesse e da come l'informazione può propagarsi**, la geometria può essere incorporata nel modello. La mesh per il meteo, le articolazioni del corpo e le relazioni fra documenti sono modi diversi di fornire un prior relazionale al calcolo delle rappresentazioni.

Un prior pertinente può aiutare la generalizzazione e limitare combinazioni non utili; non garantisce automaticamente l'assenza di overfitting o di incertezza. Le sfide citate nella slide 56, fra cui scalabilità, over-smoothing, interpretabilità e grafi che cambiano, restano aspetti da considerare nella scelta del modello.

## 23. Riepilogo delle forme e delle formule (slide 18-28)

| Oggetto | Formula o forma | Significato |
|---|---|---|
| Ingresso | $X\in\mathbb R^{N\times F_0}$ | Una riga per nodo |
| Adiacenza | $A\in\mathbb R^{N\times N}$ | Connessioni del grafo |
| Self-loop | $\widehat A=A+I$ | Contributo del nodo stesso |
| Gradi aumentati | $\widehat D_{ii}=\sum_j\widehat A_{ij}$ | Normalizzazione della nuova adiacenza |
| Aggregazione GCN | $\widehat S=\widehat D^{-1/2}\widehat A\widehat D^{-1/2}$ | Combinazione strutturale dei nodi |
| Proiezione | $W^{(\ell)}\in\mathbb R^{F_\ell\times F_{\ell+1}}$ | Feature in ingresso e uscita |
| Strato | $H^{(\ell+1)}=\sigma(\widehat SH^{(\ell)}W^{(\ell)})$ | Una nuova rappresentazione per nodo |
| Laplaciano di partenza | $L=I-D^{-1/2}AD^{-1/2}$ | Operatore spettrale sul grafo iniziale |
| Base spettrale | $L=U\Lambda U^\top$ | Autovettori e autovalori |
| Trasformata e inversa | $\widehat x=U^\top x$, $x=U\widehat x$ | Cambio di coordinate e ricostruzione |
| Filtro spettrale | $y=Ug(\Lambda)U^\top x$ | Una scala per ogni frequenza |
| Riscalatura | $\widetilde L=2L/\lambda_{\max}-I$ | Spettro nell'intervallo di Chebyshev |
| Identità polinomiale | $UT_k(\widetilde\Lambda)U^\top=T_k(\widetilde L)$ | Eliminazione esatta degli autovettori |
| Primo grado vincolato | $\theta(I+S)x$ | Due termini con coefficienti opposti |
| Rinormalizzazione | $I+S\longmapsto\widehat S$ | Cambio di operatore, non identità |

## 24. Errata e precisazioni essenziali (slide 5-28)

| Riferimento | Precisazione |
|---|---|
| Slide 5 e richiamo orale | Convoluzione e correlazione differiscono per il ribaltamento del kernel; la scelta dell'origine non è la distinzione fondamentale. |
| Slide 6-9 | Il reticolo a quattro vicini non coincide automaticamente con tutti gli offset di un kernel $3\times3$. |
| Slide 13-14 | La località è topologica; assenza di una griglia non significa assenza di cammini o distanze sul grafo. |
| Slide 13 | Le uscite per nodo sono equivarianti alla rinumerazione coerente; l'invarianza globale richiede una riduzione appropriata. |
| Slide 18-19 | Un grafo non orientato ha adiacenza simmetrica; ciò non impone un cammino unico. I pesi non sono necessariamente probabilità. |
| Slide 20-21 e terminologia orale | L'operatore della GCN è un'adiacenza normalizzata, distinta dal Laplaciano $I-S$. |
| Slide 21 | La dimensione delle feature di ingresso è quella dello strato precedente; gli indici devono essere coerenti con $W$. |
| Slide 22 | La formula con numeratore 1 vale per archi binari; nel caso pesato va mantenuto il peso dell'arco. La normalizzazione simmetrica non è una softmax. |
| Slide 23 | Aumentano le dipendenze raggiungibili, non necessariamente gli archi. Componenti disconnesse restano separate. |
| Slide 24-25 | La base ortonormale segue dalla simmetria; una SVD generale non ha necessariamente la stessa matrice sui due lati. |
| Slide 24-27 | L'identità polinomiale matriciale vale su qualsiasi segnale; un filtro spettrale generale non è sempre un polinomio finito. |
| Slide 26 | La limitatezza di $T_k$ vale sull'intervallo $[-1,1]$; il grado $K$ comporta $K+1$ termini. |
| Slide 27 | L'approssimazione riguarda la scelta del filtro polinomiale; l'eliminazione di $U$ è esatta per quel filtro. |
| Slide 28 | Mantenere $T_0,T_1$ significa grado 1. Il vincolo sui coefficienti e la rinormalizzazione sono scelte ulteriori. |
| Slide 28 | $I+S$ e $\widehat S$ non sono uguali. Il controllo spettrale della propagazione non garantisce da solo stabilità dell'intera rete. |
