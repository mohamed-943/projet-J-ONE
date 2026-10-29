# Notions théoriques — STT avec un Transformer

Ce document sert à la fois de **fiche de révision** et de **base pour la partie théorique du rapport**. Chaque notion est présentée en quatre parties :

- **Explication :** ce que c'est, avec des mots simples et un exemple ;
- **Où on l'a appliquée :** le fichier ou le notebook du projet ;
- **Pourquoi on l'a appliquée :** le problème observé dans nos données ou notre code, et comment la notion le résout (avec les chiffres) ;
- **Code :** le morceau de code retenu, avec ses commentaires.

Le journal (`JOURNAL_PROJET.md`) raconte le projet dans l'ordre chronologique. Ce document regroupe les mêmes éléments **par notion**.

---

## 1. Échantillonnage

**Explication.** Un micro capte une onde de pression. La conversion analogique-numérique la mesure à intervalles réguliers : chaque mesure est un **échantillon**. La **fréquence d'échantillonnage** est le nombre de mesures par seconde. Exemple : 3 s à 16 kHz donnent 3 × 16 000 = 48 000 échantillons. Elle ne doit pas être confondue avec la **fréquence du signal**, qui compte les oscillations par seconde (440 Hz pour la note la).

**Où on l'a appliquée.**
- `00_signal_audio.ipynb` : une sinusoïde de 440 Hz. Un extrait de 200 échantillons dure 0,0125 s et contient 5,5 cycles.
- `stt_01_exploration.ipynb`, points 3 et 4.

**Pourquoi.** L'EDA a montré que nos clips mélangent **trois fréquences d'échantillonnage** : 32 000, 44 100 et 48 000 Hz. Une même seconde de parole ne donnait donc pas le même nombre d'échantillons selon le clip. **Solution :** tout ramener à **16 000 Hz**.

**Code** (`00_signal_audio.ipynb`, puis `stt/audio.py`) :

```python
# Signal artificiel : instants de mesure, puis valeur de la sinusoïde à chaque instant
sample_rate = 16000                                  # 16 000 échantillons par seconde
duration = 3                                         # durée en secondes
t = np.arange(sample_rate * duration) / sample_rate  # 0, 1/16000, 2/16000, ...
audio = np.sin(2 * np.pi * 440 * t)                  # sinusoïde à 440 Hz

# Dans le projet : la fréquence cible commune
TARGET_SR = 16000
```

---

## 2. Théorème de Nyquist et aliasing

**Explication.** Un signal échantillonné à *f* Hz ne peut représenter que les fréquences inférieures à ***f* / 2**, la **fréquence de Nyquist**. Une fréquence plus haute se replie et devient une **fausse fréquence** : c'est l'**aliasing**. Pour réduire la fréquence d'échantillonnage sans aliasing, on supprime d'abord les fréquences trop hautes avec un **filtre passe-bas**.

**Où on l'a appliquée.** `stt_02_pretraitement.ipynb`, étape 2, et la fonction `load_audio` de `stt/audio.py`.

**Pourquoi.**
- **16 kHz suffit pour la parole.** La limite de Nyquist est alors de 8 kHz. Les données sont 2 à 3 fois plus légères qu'en 32 ou 48 kHz : un clip de 5 s compte 80 000 échantillons au lieu de 240 000.
- **Le filtre est indispensable.** Nos clips à 48 kHz contiennent des fréquences jusqu'à 24 kHz. Sans filtre, tout ce qui se situe entre 8 et 24 kHz se replierait en fréquences parasites.

**Code** (`stt/audio.py`) :

```python
def load_audio(path):
    # mp3 -> tenseur 1D à 16 kHz
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim == 2:                  # par sécurité : stéréo -> mono
        audio = audio.mean(axis=1)
    waveform = torch.from_numpy(audio)
    if sr != TARGET_SR:
        # resample applique d'abord un filtre passe-bas (anti-aliasing),
        # puis réduit le nombre d'échantillons
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=TARGET_SR)
    return waveform
```

---

## 3. FFT (transformée de Fourier rapide)

**Explication.** La FFT décompose un morceau de signal en une somme de sinusoïdes et donne l'**intensité de chaque fréquence**. Analogie : la forme d'onde, c'est le son d'un accord ; la FFT, c'est la liste de ses notes. La **résolution fréquentielle** est égale à la fréquence d'échantillonnage divisée par la taille de la fenêtre (16 000 / 400 = 40 Hz).

**Où on l'a appliquée.** `00_signal_audio.ipynb`, expérience 2 :
- pour une sinusoïde de 440 Hz, un pic unique à 440 Hz ;
- pour le mélange 440 + 1 000 Hz, deux pics de hauteurs 200 et 100.

**Pourquoi.** La forme d'onde du mélange était illisible. La FFT sépare les fréquences. C'est la brique de base du spectrogramme, qui est l'entrée de notre modèle.

**Code** (`00_signal_audio.ipynb`) :

```python
window_size = 400                        # 25 ms à 16 kHz
window = audio[:window_size]

spectrum = np.abs(np.fft.rfft(window))   # magnitude de chaque fréquence
freqs = np.fft.rfftfreq(window_size, 1 / sample_rate)   # fréquence (Hz) de chaque case
```

---

## 4. STFT et spectrogramme

**Explication.** Le **STFT** (*Short-Time Fourier Transform*) applique une FFT sur des **fenêtres courtes** qui glissent le long du signal. On obtient une matrice (temps × fréquences) : le **spectrogramme**.
- Il y a un **compromis temps / fréquence** : une grande fenêtre est précise en fréquence mais floue dans le temps.
- La **fenêtre de Hann** atténue les bords de chaque fenêtre pour éviter les fréquences parasites.
- Nombre d'instants = 1 + ⌊échantillons / pas⌋.

**Où on l'a appliquée.** `00_signal_audio.ipynb` (STFT écrit à la main) et `stt/audio.py` (`MelSpectrogram`, avec une fenêtre de 25 ms et un pas de 10 ms).

**Pourquoi.** La parole change sans cesse. Une seule FFT dirait **quelles** fréquences sont présentes, mais pas **quand**. 25 ms / 10 ms sont les réglages de référence, et ce sont aussi ceux de Whisper, ce qui rend la comparaison plus juste. Exemple : 74 496 échantillons donnent 1 + ⌊74 496 / 160⌋ = **466 instants**.

**Code** (`00_signal_audio.ipynb`, STFT écrit à la main) :

```python
def stft_magnitude(signal, window_size, hop):
    frames = []
    # On fait glisser la fenêtre de `hop` échantillons à chaque pas
    for start in range(0, len(signal) - window_size + 1, hop):
        window = signal[start:start + window_size]
        frames.append(np.abs(np.fft.rfft(window)))   # une FFT par fenêtre
    return np.array(frames)  # forme : (nombre de fenêtres, nombre de fréquences)
```

---

## 5. Échelle Mel et logarithme

**Explication.**
- L'oreille distingue finement les graves et grossièrement les aigus. L'**échelle Mel** regroupe donc les fréquences en bandes **étroites dans les graves** et **larges dans les aigus**.
- L'oreille perçoit le volume de façon **logarithmique**, et le log comprime les grands écarts de valeurs. On écrit `log(x + 1e-6)`, car log(0) vaut −∞.

**Où on l'a appliquée.** `stt_02_pretraitement.ipynb`, étape 3, et la fonction `log_mel` de `stt/audio.py` (80 bandes).

**Pourquoi.**
- On passe de 201 fréquences à 80 bandes pertinentes pour la parole.
- On a observé un minimum d'exactement −13,8 = log(1e‑6), donc des bandes d'énergie **nulle** (le silence ajouté au début des fichiers mp3). Sans le `1e-6`, on aurait obtenu −∞, et l'entraînement aurait planté.

**Code** (`stt/audio.py`) :

```python
N_FFT = 400        # fenêtre de 25 ms
HOP_LENGTH = 160   # pas de 10 ms
N_MELS = 80

# Créé une seule fois, réutilisé pour tous les clips
_mel_transform = T.MelSpectrogram(
    sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)

mel = _mel_transform(waveform)               # (80, instants)
features = torch.log(mel + 1e-6).T           # log, puis (instants, 80)
```

---

## 6. Normalisation des entrées

**Explication.** Un réseau apprend mieux quand ses entrées sont **centrées** (moyenne 0) et **réduites** (écart-type 1). Sinon, les gradients sont déséquilibrés et l'apprentissage est lent et instable. Formule : (x − moyenne) / (écart-type + 1e‑5).

**Où on l'a appliquée.** `stt_02_pretraitement.ipynb`, étape 4, et la fonction `log_mel` de `stt/audio.py` : normalisation **par clip** et **par bande Mel**.

**Pourquoi.**
- Le log-Mel brut avait une moyenne de −7,8 et un écart-type de 5,6. Après normalisation, la moyenne est ≈ 0 et l'écart-type ≈ 1.
- Le choix « par clip » compense les différences de **micro** et de **volume** entre les enregistrements. Ce sera aussi le cas avec le **micro du robot**.

**Code** (`stt/audio.py`) :

```python
mean = features.mean(dim=0, keepdim=True)   # une moyenne par bande : (1, 80)
std = features.std(dim=0, keepdim=True)     # un écart-type par bande : (1, 80)
# Le 1e-5 évite une division par 0 si une bande est constante dans tout le clip
return (features - mean) / (std + 1e-5)
```

---

## 7. Le paramètre `dim` dans PyTorch

**Explication.** Dans une réduction (moyenne, somme…), `dim` désigne la dimension qui **disparaît**. Elle devient de taille 1 avec `keepdim=True`. Sur un tenseur (instants, bandes) :
- `mean(dim=0)` donne une moyenne **par bande**, de forme `(1, 80)` ;
- `mean(dim=1)` donne une moyenne **par instant**, de forme `(466, 1)`.

**Où on l'a appliquée.** La fonction `log_mel` de `stt/audio.py`.

**Pourquoi.** Chaque bande (graves, aigus…) a son propre niveau d'énergie, qu'il faut ramener à 0 séparément. Avec `dim=1`, on mélangerait les graves et les aigus d'un même instant, et on effacerait ce qui distingue un « a » d'un « s ».

**Code** (`stt/audio.py`) :

```python
# dim=0 : on « écrase » les instants -> une valeur par bande Mel
mean = features.mean(dim=0, keepdim=True)   # (466, 80) -> (1, 80)
```

---

## 8. Normalisation du texte et vocabulaire

**Explication.** Le modèle produit un score par symbole de son **vocabulaire**. Un symbole rarement vu ne peut pas être appris. Normaliser le texte réduit le vocabulaire aux symboles utiles. La **normalisation Unicode NFC** fusionne les accents « décomposés » (`e` suivi de `´` devient `é`).

**Où on l'a appliquée.** `stt_01_exploration.ipynb`, point 2, et `stt/text.py`.

**Pourquoi.** L'EDA a trouvé **178 caractères différents** : des majuscules (qui ne s'entendent pas), deux apostrophes, des accents décomposés et des symboles vus une seule fois. Après normalisation, il en reste **49**, on n'a écarté que 1,4 % des phrases, et la sortie du modèle compte **50 scores** (49 + le blank).

**Code** (`stt/text.py`) :

```python
# Caractères autorisés : lettres françaises + espace + ponctuation retenue
ALLOWED = set("abcdefghijklmnopqrstuvwxyz" "àâæçéèêëîïôœùûüÿ" " '-.,?!")

# Vocabulaire : le blank du CTC en position 0, puis les 49 caractères triés
BLANK = "_"
VOCAB = [BLANK] + sorted(ALLOWED)
CHAR_TO_ID = {char: i for i, char in enumerate(VOCAB)}


def normalize(text):
    text = unicodedata.normalize("NFC", text)        # recolle les accents décomposés
    text = text.lower()
    text = re.sub(r"[’‘ʼ´]", "'", text)              # une seule apostrophe
    text = re.sub(r"[;:…«»()\"—–]", " ", text)       # ponctuation secondaire -> espace
    text = re.sub(r" +", " ", text).strip()
    return text


def encode(text):
    # Texte -> tenseur d'indices (la cible de la loss CTC)
    return torch.tensor([CHAR_TO_ID[char] for char in text], dtype=torch.long)
```

---

## 9. Le CTC (Connectionist Temporal Classification)

**Explication.** Un clip de 5 s compte 500 instants, mais sa phrase ne fait que 60 caractères, et on ne sait pas quel instant correspond à quelle lettre. Le **CTC** entraîne le modèle **sans cet alignement** : il additionne les probabilités de **tous les alignements possibles** qui donnent la bonne phrase.
- **Le blank** (`_`) signifie « rien de nouveau ici ». Pour lire la sortie, on fusionne les symboles identiques qui se suivent, puis on supprime les blanks : `b b o _ n n _ n e _` donne « bonne ».
- **La contrainte de longueur :** le nombre d'instants doit être ≥ nombre de caractères + nombre de lettres doublées.

**Où on l'a appliquée.** `stt_01_exploration.ipynb`, point 5, et `stt/text.py` (le blank a l'indice 0, la valeur par défaut de `nn.CTCLoss`).

**Pourquoi.**
- Common Voice ne donne jamais la position de chaque lettre dans le temps.
- La contrainte de longueur fixe une limite à l'architecture. Nos phrases demandent jusqu'à **22,6 instants par seconde**. Une réduction ÷ 4 (25 instants/s) passe, alors qu'une réduction ÷ 8 rendrait **44 %** des clips impossibles.

**Code** (`stt_01_exploration.ipynb`, mesure de la contrainte) :

```python
def count_repeats(text):
    # Compte les paires de caractères identiques qui se suivent ("nn", "ss", "ll"...)
    return sum(a == b for a, b in zip(text, text[1:]))

# Instants minimum nécessaires pour le CTC : caractères + blanks obligatoires
df["ctc_min"] = df["text"].str.len() + df["text"].map(count_repeats)
# Instants nécessaires par seconde d'audio
df["ctc_per_sec"] = df["ctc_min"] / df["duration"]
```

---

## 10. Dataset, DataLoader, padding et masque

**Explication.**
- Un **Dataset** répond aux questions « combien d'exemples ? » (`__len__`) et « donne-moi l'exemple n° *i* » (`__getitem__`).
- Un **DataLoader** mélange les exemples et les regroupe en batchs.
- Les clips plus courts sont complétés par des zéros (**padding**), et on garde les **vraies longueurs** à côté.

**Où on l'a appliquée.** `stt/dataset.py`.

**Pourquoi.**
- On ne peut pas prétraiter 12 982 clips à la main, et on ne veut pas tous les garder en mémoire.
- Les clips ont des longueurs différentes (de 389 à 757 instants dans un batch observé).
- Les vraies longueurs servent au CTC et au masque d'attention.
- Le clip le plus court de ce batch contenait **49 % de padding**. Piste d'optimisation : le *bucketing*.

**Code** (`stt/dataset.py`) :

```python
class CommonVoiceDataset(Dataset):
    def __init__(self, table):
        # Appelé une fois, à la création : on mémorise le tableau des clips
        self.table = table

    def __len__(self):
        # Nombre de clips
        return len(self.table)

    def __getitem__(self, i):
        # Prétraite le clip n° i à la demande et renvoie (features, cible)
        row = self.table.iloc[i]
        features = log_mel(load_audio(row["audio_path"]))   # (instants, 80)
        target = encode(row["text"])                          # (nombre de caractères,)
        return features, target


def collate_batch(batch):
    # batch = liste de paires (features, cible) renvoyées par __getitem__
    features_list, targets_list = zip(*batch)
    # Vraies longueurs, avant padding (nécessaires au CTC et au masque d'attention)
    feature_lengths = torch.tensor([f.shape[0] for f in features_list])
    target_lengths = torch.tensor([t.shape[0] for t in targets_list])
    # Complète avec des 0 jusqu'à la longueur du plus long élément du batch
    features = pad_sequence(features_list, batch_first=True)   # (batch, instants max, 80)
    targets = pad_sequence(targets_list, batch_first=True)     # (batch, caractères max)
    return features, feature_lengths, targets, target_lengths
```

---

## 11. La convolution

**Explication.** Une convolution fait glisser un petit **filtre** (une grille de poids **apprise**) sur les données. À chaque position, on multiplie case par case, puis on additionne. Le résultat mesure à quel point la zone ressemble au motif du filtre.
- *Exemple en 1D :* avec le signal `[1, 2, 3, 4, 5, 6]` et le filtre `[1, 0, -1]`, on obtient `[−2, −2, −2, −2]`. Ce filtre réagit aux **changements**.
- **Pas (*stride*)** : sortie = ⌊(entrée − taille du filtre) / pas⌋ + 1.
- **En 2D**, sur le spectrogramme (temps × fréquences), un filtre 3 × 3 apprend des **motifs locaux du son**. Avec *C* filtres, on obtient *C* **canaux**.

| | Instants | Bandes |
|---|---|---|
| Entrée | 466 | 80 |
| Après la 1re convolution | ⌊(466 − 3) / 2⌋ + 1 = 232 | 39 |
| Après la 2e convolution | ⌊(232 − 3) / 2⌋ + 1 = 115 | 19 |

**Où on l'a appliquée.** La classe `ConvSubsampling` de `stt/model.py`.

**Pourquoi.**
- **Le coût de l'attention croît avec le carré du nombre d'instants** : 466² ≈ 217 000 scores contre 115² ≈ 13 000, soit **16 fois moins**.
- **La contrainte du CTC est respectée** : il faut 63 instants, et il en reste 115.
- **Vérification sur un batch réel :** 663 → 165 instants.

**Code** (`stt/model.py`) :

```python
def conv_out_length(length):
    """Longueur de sortie d'une convolution de filtre 3, pas 2 : ⌊(entrée − 3) / 2⌋ + 1."""
    return (length - 3) // 2 + 1


class ConvSubsampling(nn.Module):
    def __init__(self, n_mels, channels, d_model):
        super().__init__()  # obligatoire : initialise la partie nn.Module
        self.conv = nn.Sequential(
            # 1re convolution : le spectrogramme (1 canal) passe dans `channels` filtres 3×3.
            # Le pas de 2 divise le temps ET les fréquences par ~2 (466×80 -> 232×39).
            nn.Conv2d(1, channels, kernel_size=3, stride=2),
            nn.ReLU(),
            # 2e convolution : combine les motifs détectés par la 1re (232×39 -> 115×19).
            nn.Conv2d(channels, channels, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        # Nombre de bandes de fréquence restant après les deux convolutions : 80 -> 39 -> 19
        freq_out = conv_out_length(conv_out_length(n_mels))
        # Chaque instant est décrit par channels × freq_out valeurs -> projeté vers d_model
        self.linear = nn.Linear(channels * freq_out, d_model)

    def forward(self, x, lengths):
        x = x.unsqueeze(1)                       # (B, 1, T, 80) : ajoute la dimension « canal »
        x = self.conv(x)                         # (B, C, T', 19), avec T' ≈ T / 4
        b, c, t, f = x.shape
        x = x.permute(0, 2, 1, 3)                # (B, T', C, 19) : le temps en 2e position
        x = x.reshape(b, t, c * f)               # (B, T', C×19) : chaque instant = un vecteur
        x = self.linear(x)                       # (B, T', d_model)
        # Les vraies longueurs subissent la même réduction
        lengths = conv_out_length(conv_out_length(lengths))
        return x, lengths
```

---

## 12. La fonction d'activation ReLU

**Explication.** ReLU(x) = max(0, x) : les valeurs négatives deviennent 0. Exemple : `[−2, 0.5, −1, 3]` devient `[0, 0.5, 0, 3]`.

Une convolution ou un `nn.Linear` est une opération **linéaire**. Or deux opérations linéaires enchaînées n'en font qu'une : *g*(*f*(x)) = 3 × 2x = 6x, et de même W₂(W₁x) = (W₂W₁)x. Une **non-linéarité** placée entre les deux casse cette équivalence.

**Où on l'a appliquée.** `ConvSubsampling`, avec une ReLU après chaque convolution.

**Pourquoi.**
- Sans ReLU, la 2e convolution ne servirait à rien : les deux reviendraient à **une seule**. La 2e couche ne pourrait pas **combiner** les motifs simples de la 1re (« énergie qui monte », « énergie dans les aigus ») en motifs plus riches (« début d'un s »).
- ReLU est très rapide, et sa dérivée vaut 1 pour les valeurs positives, ce qui laisse bien passer les gradients.

**Code** (`stt/model.py`) :

```python
nn.Conv2d(1, channels, kernel_size=3, stride=2),
# ReLU : remplace les valeurs négatives par 0 (non-linéarité).
# Sans elle, deux convolutions de suite seraient équivalentes à UNE seule
# opération linéaire : empiler des couches ne servirait à rien.
nn.ReLU(),
```

---

## 13. L'encodage des positions (sinusoïdal)

**Explication.** L'attention **ignore l'ordre** des positions. On ajoute donc à chaque vecteur un motif qui dépend de sa position :

> PE(pos, 2i) = sin(pos / 10000^(2i/d)) et PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

*Analogie :* une horloge à plusieurs aiguilles, rapides et lentes. Leur combinaison est **unique pour chaque position**. Exemple : la position 0 donne `[0, 1, 0, 1, …]`, la position 1 donne `[0,84 ; 0,54 ; …]`. Le motif n'est **pas appris** (`register_buffer`).

**Où on l'a appliquée.** La classe `SinusoidalPositionalEncoding` de `stt/model.py`.

**Pourquoi.**
- « Le chat mange » et « mange le chat » contiennent les mêmes sons, mais pas dans le même ordre.
- **Choix du sinusoïdal :** le test spontané va jusqu'à **83 s**, alors que l'entraînement s'arrête à 11 s. Un encodage appris ne connaîtrait pas ces positions, alors que le sinusoïdal est défini partout.
- **Critère pour en changer :** si le WER se dégrade sur les clips de plus de 30 s, on passera au RoPE (encodage relatif).

**Code** (`stt/model.py`) :

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        # max_len : 10 000 instants = 400 s, au-dessus du clip le plus long (83 s ≈ 2 075 instants)
        super().__init__()
        # Numéros de position 0, 1, 2, ..., en colonne : (max_len, 1)
        position = torch.arange(max_len).unsqueeze(1)
        # Une fréquence par paire de dimensions : rapide pour les premières, lente pour les dernières.
        # C'est 1 / 10000^(2i / d_model), calculé avec exp et log pour éviter de trop grands nombres.
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)   # dimensions paires : sinus
        pe[:, 1::2] = torch.cos(position * div_term)   # dimensions impaires : cosinus
        # « buffer » : fait partie du module (suit .to("cuda")), mais n'est PAS appris.
        self.register_buffer("pe", pe)

    def forward(self, x):
        # (T', d_model) s'additionne à (B, T', d_model) par broadcasting
        return x + self.pe[: x.shape[1]]
```

---

## 14. L'attention (scaled dot-product) et le masque du padding

**Explication.** Chaque instant produit trois vecteurs : une **requête Q** (« ce que je cherche »), une **clé K** (« ce que je contiens ») et une **valeur V** (« ce que je transmets »).
1. Q·Kᵀ / √d_k donne les **scores** (T × T). La division par √d_k évite des scores trop grands, qui rendraient le softmax trop « pointu » et les gradients trop faibles.
2. Le **softmax** transforme chaque ligne en poids qui somment à 1.
3. La **moyenne pondérée** des V donne à chaque instant un résumé du contexte.

**Le masque** remplace par −∞ les scores des colonnes de padding, qui reçoivent alors un poids exactement nul (e^(−∞) = 0).

**Le test de référence** compare notre version à celle de PyTorch avec les mêmes entrées.

**Où on l'a appliquée.** Les fonctions `make_padding_mask` et `scaled_dot_product_attention` de `stt/model.py`.

**Pourquoi.**
- Pour reconnaître un son, il faut son **contexte**.
- Nos batchs contiennent jusqu'à **49 % de padding**. Sans masque, le résultat d'un clip dépendrait de la longueur des **autres** clips du batch.
- **Test :** les colonnes de padding reçoivent un poids de 0,00, chaque ligne somme à 1, et le résultat est « Identique à PyTorch : True ».
- **Compromis entre apprentissage et standards professionnels :** on écrit l'attention pour la comprendre et pouvoir déboguer, on la vérifie, puis on **entraîne avec la version optimisée de PyTorch** (FlashAttention).

**Code** (`stt/model.py`) :

```python
def make_padding_mask(lengths, max_len):
    """Masque booléen (B, T) : True pour un vrai instant, False pour du padding."""
    positions = torch.arange(max_len, device=lengths.device)   # (T,)  : 0, 1, 2, ...
    # Comparaison (1, T) < (B, 1) -> (B, T) par broadcasting
    return positions[None, :] < lengths[:, None]


def scaled_dot_product_attention(q, k, v, mask):
    d_k = q.shape[-1]
    # Score de chaque paire (instant i, instant j) : à quel point i doit regarder j.
    # (B, T, d_k) @ (B, d_k, T) -> (B, T, T). Division par √d_k pour éviter des scores
    # trop grands, qui rendraient le softmax trop « pointu » et les gradients trop faibles.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    # Masque : aucun instant ne doit regarder le padding.
    # mask[:, None, :] a la forme (B, 1, T) : on masque des COLONNES (les instants regardés).
    # ~ inverse le masque : True là où il y a du padding -> score remplacé par -∞.
    scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
    # Softmax sur la dernière dimension : les poids de chaque ligne somment à 1
    # (le padding reçoit exactement 0).
    weights = torch.softmax(scores, dim=-1)
    # Moyenne pondérée des valeurs V : chaque instant reçoit un résumé du contexte.
    return weights @ v, weights
```

---

## 15. La multi-head attention

**Explication.** On réalise **h attentions en parallèle** (les têtes), chacune sur d_k = d_model / h dimensions. Par exemple, 144 dimensions divisées en 4 têtes donnent 36 dimensions par tête. Une seule couche produit Q, K et V, on découpe en têtes `(B, h, T, d_k)`, on calcule l'attention, on recolle les têtes, puis une couche de sortie mélange leurs informations.

**Où on l'a appliquée.** La classe `MultiHeadSelfAttention` de `stt/model.py`. L'option `use_pytorch_kernel` permet de choisir entre notre version et la version optimisée.

**Pourquoi.**
- Une seule tête doit faire un **compromis** entre regarder les sons voisins et regarder un contexte plus large. Plusieurs têtes peuvent se spécialiser.
- **Test :** le résultat est identique à `nn.MultiheadAttention` (avec les mêmes poids copiés) et à la version optimisée.
- **Piège rencontré :** le paramètre `key_padding_mask` de PyTorch utilise la convention **inverse** (True = à ignorer).

**Code** (`stt/model.py`) :

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_pytorch_kernel=True):
        super().__init__()
        # Chaque tête reçoit d_model / n_heads dimensions : la division doit tomber juste.
        assert d_model % n_heads == 0, "d_model doit être divisible par n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # Une seule couche produit Q, K et V d'un coup (3 × d_model sorties) : plus rapide.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Recombine les sorties des têtes
        self.out = nn.Linear(d_model, d_model)
        # True : version optimisée de PyTorch (entraînement). False : notre version (vérification).
        self.use_pytorch_kernel = use_pytorch_kernel

    def forward(self, x, mask):
        B, T, d_model = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)          # trois tenseurs (B, T, d_model)
        # Découpage en têtes : (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        q = q.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # Masque de forme (B, 1, T) : la même règle s'applique à toutes les têtes.
        head_mask = mask[:, None, :]
        if self.use_pytorch_kernel:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=head_mask[:, None, :, :])
        else:
            out, _ = scaled_dot_product_attention(q, k, v, head_mask)
        # Recollage des têtes : (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, d_model)
        out = out.transpose(1, 2).reshape(B, T, d_model)
        return self.out(out)
```

---

## 16. Le feed-forward, GELU et dropout

**Explication.**
- Le **feed-forward** est un petit réseau appliqué **à chaque instant séparément** : Linear (élargit de d_model à d_ff, en général 4 × d_model), puis activation, puis Linear (revient à d_model). L'attention **mélange** l'information entre instants ; le feed-forward la **transforme** pour chaque instant.
- **GELU** est une version « douce » de ReLU : les petites valeurs négatives ne sont pas coupées net à 0. C'est le standard des Transformers (Whisper, BERT, GPT).
- Le **dropout** met **au hasard** à 0 une partie des valeurs (ici 10 %) **pendant l'entraînement seulement**. Le modèle ne peut pas compter sur un seul neurone, ce qui limite l'overfitting.
- **`.train()` et `.eval()`** basculent le module entre le mode entraînement (dropout actif) et le mode évaluation (dropout désactivé).

**Où on l'a appliquée.** La classe `FeedForward` de `stt/model.py`, ainsi que le dropout autour de chaque sous-bloc de `TransformerBlock`.

**Pourquoi.**
- **GELU :** la différence avec ReLU est faible, mais c'est le standard des Transformers, et elle est toujours légèrement favorable dans la littérature.
- **Dropout de 0,1 :** l'EDA (point 6) a montré qu'on n'a que **275 locuteurs**, et que les 10 plus actifs enregistrent 44 à 54 % des clips. Le risque d'overfitting est réel.
- **`.eval()` dans le test de référence :** sans lui, les zéros aléatoires du dropout auraient différé entre les deux versions.

**Code** (`stt/model.py`) :

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),    # élargit : d_model -> d_ff (en général 4 × d_model)
            nn.GELU(),                   # non-linéarité (sinon deux Linear = un seul)
            nn.Dropout(dropout),         # met au hasard 10 % des valeurs à 0 (entraînement seulement)
            nn.Linear(d_ff, d_model),    # revient à d_model (indispensable pour le résiduel)
        )

    def forward(self, x):
        return self.net(x)
```

---

## 17. Le bloc Transformer : connexions résiduelles, LayerNorm, pre-norm

**Explication.**
- La **connexion résiduelle** `x + sous_bloc(x)` conserve l'information d'origine, et les gradients remontent directement vers les couches précédentes. Elle n'est possible que si le sous-bloc **garde la taille `d_model`**.
- La **LayerNorm** normalise **chaque instant** (moyenne 0, écart-type 1 sur ses `d_model` valeurs), avec une échelle et un décalage appris.
- **Pre-norm :** `x + sous_bloc(LayerNorm(x))`, la version moderne. La **post-norm** `LayerNorm(x + sous_bloc(x))` est celle du Transformer original de 2017.

**Où on l'a appliquée.** La classe `TransformerBlock` de `stt/model.py`.

**Pourquoi.**
- **Pre-norm :** elle est nettement plus stable à l'entraînement. Le chemin résiduel n'est jamais normalisé, donc les gradients circulent mieux. C'est précieux pour un premier modèle entraîné **de zéro** avec un budget limité (30 h de GPU par semaine), où un entraînement qui diverge coûte des heures. C'est aussi le choix de Whisper et des modèles récents.
- **Test :** le résultat est identique à `nn.TransformerEncoderLayer(norm_first=True, activation="gelu")`, avec les mêmes poids copiés et en mode `.eval()`.

**Code** (`stt/model.py`) :

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, use_pytorch_kernel=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, n_heads, use_pytorch_kernel)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 1. Attention : chaque instant puise de l'information dans les autres.
        #    Connexion résiduelle (x + ...) : l'information d'origine est conservée
        #    et les gradients circulent directement vers les couches précédentes.
        x = x + self.dropout(self.attention(self.norm1(x), mask))
        # 2. Feed-forward : chaque instant est transformé séparément.
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x
```

---

## 18. Le modèle complet : output head, log_softmax, LayerNorm finale, taille du modèle

**Explication.**
- **Output head :** une couche `Linear(d_model → 50)` qui donne, à chaque instant, un **score** (*logit*) pour chacun des 50 symboles (49 caractères + le blank).
- **log_softmax :** elle transforme les scores en **log-probabilités**. Le softmax donne des probabilités qui somment à 1, puis on en prend le log. `nn.CTCLoss` attend ce format. Faire les deux opérations en une seule est **numériquement stable** : on évite de calculer log(0) quand une probabilité est minuscule.
- **LayerNorm finale :** en pre-norm, le chemin résiduel ne passe jamais par une LayerNorm, donc la sortie du dernier bloc n'est pas normalisée. On en ajoute une avant l'output head.
- **`nn.ModuleList` :** une liste de modules dont PyTorch **suit les paramètres**. Avec une liste Python ordinaire, l'optimizer ne verrait pas les poids des blocs, et ils ne seraient jamais entraînés.
- **Taille du modèle :** le nombre de paramètres doit être proportionné à la quantité de données. Un modèle trop gros **mémorise** les données (overfitting), un modèle trop petit n'a pas assez de capacité.

**Où on l'a appliquée.** La classe `SpeechTransformer` de `stt/model.py`.

**Pourquoi.**
- **La taille S (3,6 M de paramètres).** Nous avons 19 h et 275 locuteurs. À titre de comparaison, Whisper tiny compte 39 M de paramètres pour 680 000 h de données, et Conformer-S environ 10 M pour 960 h.
- **Test sur un vrai batch :**
  - l'entrée `(8, 627, 80)` donne une sortie `(8, 156, 50)` ;
  - la contrainte du CTC est respectée (par exemple 94 instants pour 55 caractères) ;
  - les probabilités somment à 1.

**Code** (`stt/model.py`) :

```python
    def forward(self, features, feature_lengths):
        # Étape 1 — sous-échantillonnage : le temps est divisé par ~4
        x, lengths = self.subsampling(features, feature_lengths)       # (B, T', d_model)
        # Étape 2 — ajout de la position de chaque instant
        x = self.dropout(self.positional_encoding(x))                  # (B, T', d_model)
        # Masque du padding, calculé une seule fois pour tous les blocs
        mask = make_padding_mask(lengths, x.shape[1])                  # (B, T')
        # Étape 3 — les blocs Transformer, l'un après l'autre
        for block in self.blocks:
            x = block(x, mask)                                         # (B, T', d_model)
        # Étape 4 — normalisation finale, puis un score par symbole du vocabulaire
        x = self.final_norm(x)
        logits = self.output_head(x)                                   # (B, T', 50)
        # Étape 5 — log-probabilités : nn.CTCLoss attend log(softmax), calculé ici
        # en une seule opération numériquement stable (évite log(0))
        log_probs = logits.log_softmax(dim=-1)                         # (B, T', 50)
        return log_probs, lengths
```

---

## 19. Suivre un entraînement : époque, learning rate et métriques

**Explication du vocabulaire.**
- **Itération (*step*) :** un batch traité, c'est-à-dire forward, loss, backpropagation et mise à jour des poids.
- **Époque (*epoch*) :** **un passage complet** sur toutes les données d'entraînement. Avec 12 982 clips et des batchs de 16, une époque compte ≈ 812 itérations. On entraîne en général pendant plusieurs dizaines d'époques.
- **Learning rate (taux d'apprentissage) :** la **taille du pas** que fait l'optimizer à chaque mise à jour, dans la direction indiquée par le gradient. *Analogie :* descendre une montagne dans le brouillard. Le gradient indique la pente, et le learning rate la longueur de chaque pas.
  - **Trop grand :** on saute par-dessus la vallée. La loss oscille, explose, ou devient `nan`.
  - **Trop petit :** on avance à pas de fourmi. La loss baisse très lentement, et le quota de GPU s'épuise.
  - **En pratique :** on commence par un *warmup* (le learning rate monte doucement pendant les premières itérations, pour éviter les gros sauts au départ), puis on le fait **décroître** progressivement.

**Les métriques suivies, et comment les améliorer.**

| Métrique | À quoi elle sert | Valeur normale | Comment l'améliorer |
|---|---|---|---|
| **Loss train** (CTC) | Mesure ce que le modèle apprend sur les données d'entraînement. Elle guide l'optimizer. | Elle doit **baisser** régulièrement | Si elle stagne : ajuster le learning rate, vérifier le code (test d'overfitting), ou agrandir le modèle |
| **Loss dev** (CTC) | Mesure la même erreur sur des locuteurs **jamais vus**. Elle sert à détecter l'overfitting. | Elle doit baisser avec la loss train, en restant un peu au-dessus | Si elle remonte alors que la loss train baisse : plus de dropout, plus de données, augmentation de données, ou *early stopping* |
| **WER** (*Word Error Rate*) | **La métrique de référence de l'industrie** en STT. Elle compte les erreurs **par mot** : (substitutions + suppressions + insertions) / nombre de mots de la référence. | Plus elle est basse, mieux c'est. Whisper fait souvent < 10 % en français propre. Un modèle entraîné de zéro sur 19 h sera **beaucoup plus haut**. | Plus de données, un modèle plus grand, un **modèle de langue** au décodage, un meilleur décodage (*beam search*) |
| **CER** (*Character Error Rate*) | Même calcul que le WER, mais **par caractère**. Elle est plus fine : « marronier » au lieu de « marronnier » est une erreur entière en WER, mais une seule lettre en CER. | Toujours **plus basse** que le WER | Mêmes leviers que le WER |
| **RTF** (*Real-Time Factor*) | Le temps de calcul divisé par la durée de l'audio. **Indispensable pour le robot.** | RTF < 1 signifie plus rapide que le temps réel. Sur le robot, on vise ≪ 1. | Un modèle plus petit, un sous-échantillonnage plus fort, la quantification, un GPU |
| **Learning rate** | Indique où on en est dans le calendrier (warmup, puis décroissance) | Il suit la courbe prévue | Il se règle avant l'entraînement : c'est un hyperparamètre |
| **Norme du gradient** | La taille des gradients. Un pic signale une instabilité. | Elle reste stable | *Gradient clipping* : on plafonne la norme, par exemple à 1 |

**Lire les courbes : normal ou pas.**

| Observation | Diagnostic | Normal ? |
|---|---|---|
| Les losses train et dev baissent ensemble | Le modèle apprend et généralise | ✅ |
| La loss train baisse, la loss dev baisse puis **remonte** | **Overfitting** : le modèle mémorise | ⚠️ On garde la meilleure époque sur le dev (*early stopping*) |
| La loss train ne baisse presque pas | Le modèle n'apprend pas : learning rate, bug ou capacité | ❌ |
| `nan`, `inf` ou sauts énormes | Instabilité : learning rate trop grand, contrainte du CTC non respectée | ❌ On arrête et on corrige |
| Le modèle ne produit que des blanks au début | Phase normale du CTC | ✅ Au début seulement, ❌ si ça dure |
| La loss dev est très au-dessus de la loss train dès le départ | Écart de domaine (autres locuteurs, autres accents) | ⚠️ À analyser |

**Cas particulier : le test d'overfitting sur 10 clips.** Là, on **veut** que le modèle mémorise. La loss train doit tendre vers 0, et les 10 phrases doivent être recopiées parfaitement. S'il n'y arrive pas, il y a un bug.

**Où on l'applique.** Dans le notebook d'entraînement sur Kaggle (à venir).

**Pourquoi.** Sans ces métriques, impossible de savoir si le modèle apprend, s'il mémorise ou s'il est inutilisable. Le **WER** et le **RTF** sont aussi les deux chiffres qui permettront de **comparer notre modèle à Whisper** dans le rapport.

---

## 20. Le décodage glouton du CTC (*greedy decoding*)

**Explication.** Le modèle ne sort pas une phrase. Il sort, à **chaque instant**, une probabilité pour chacun des 50 symboles. Pour obtenir du texte, on applique les règles du CTC :
1. **argmax :** à chaque instant, on garde le symbole le plus probable ;
2. **fusion des répétitions :** plusieurs symboles identiques **à la suite** n'en font qu'un ;
3. **suppression des blanks.**

Exemple : `_ a a _ a b _ _` → (fusion) `_ a _ a b _` → (sans blanks) « aab ». Le blank placé entre les deux « a » permet de garder une **vraie double lettre** : c'est pour cela que le CTC a besoin du blank.

On parle de décodage « glouton » parce qu'on choisit le meilleur symbole instant par instant, sans revenir en arrière. L'alternative est le *beam search*, qui garde plusieurs hypothèses en parallèle. Il peut aussi utiliser un modèle de langue. Il est plus précis, mais plus lent.

**Où on l'a appliquée.** `greedy_decode` dans `stt/metrics.py`. On l'utilise dans `stt_03_overfit_10_clips.ipynb` et, plus tard, à chaque évaluation sur le dev.

**Pourquoi.**
- Sans décodage, on ne peut calculer ni WER ni CER, et on ne peut pas lire ce que le modèle « entend ». Dans le test d'overfitting, c'est grâce à lui qu'on a vu la **phase des blanks** : aux itérations 25–50, la phrase décodée était vide.
- On ne lit que les `lengths[b]` premiers instants. Les instants suivants sont du **padding** : sans cette limite, on décoderait du bruit à la fin des phrases courtes.
- Test : `_aa_ab__` donne bien « aab ». Avec une longueur de 3, on obtient « a ».

**Code** (`stt/metrics.py`) :

```python
def greedy_decode(log_probs, lengths):
    """Sortie du modèle -> phrases, avec les règles du CTC."""
    best = log_probs.argmax(dim=-1)                 # (B, T') : le symbole le plus probable à chaque instant
    sentences = []
    for b in range(best.shape[0]):
        ids, previous = [], None
        for i in best[b, :lengths[b]].tolist():     # seulement les vrais instants (pas le padding)
            if i != previous and i != 0:            # 1) fusion des répétitions  2) suppression des blanks (indice 0)
                ids.append(i)
            previous = i
        sentences.append(decode(ids))
    return sentences
```

---

## 21. La distance d'édition, le WER et le CER

**Explication.** La **distance d'édition** (ou distance de Levenshtein) est le nombre **minimum** d'opérations pour transformer la phrase de référence en phrase prédite. Il y a trois opérations :
- **substitution :** « chat » → « ch**o**t » ;
- **suppression :** « chat » → « cht » ;
- **insertion :** « chat » → « chat**s** ».

Chaque exemple coûte 1. On la calcule par **programmation dynamique** : un tableau dont la case (i, j) donne la distance entre les i premiers éléments de la référence et les j premiers de la prédiction. Chaque case se déduit de ses trois voisines, en haut, à gauche et en diagonale.

- **WER = (S + D + I) / N**, sur les **mots**, avec N le nombre de mots de la référence. Exemple : « le chat dort » → « le chien dort » donne 1 substitution sur 3 mots, soit **33 %**.
- **CER :** le même calcul sur les **caractères**.
- Le WER et le CER peuvent **dépasser 100 %**, puisque les insertions s'ajoutent sans limite.
- Sur plusieurs phrases, on **additionne** les erreurs et les longueurs avant de diviser. On ne fait pas la moyenne des pourcentages, car une phrase de 3 mots compterait autant qu'une phrase de 30.

**Où on l'a appliquée.** `edit_distance` et `error_rate` dans `stt/metrics.py`.

**Pourquoi.**
- Le WER est la **métrique de référence** en STT (notion 19). C'est lui qui permettra de comparer notre modèle à Whisper.
- La loss CTC dit si le modèle progresse, mais pas **combien de mots** il rate. Dans le test d'overfitting, une loss de 0,038 correspondait à un WER de 8,6 %, et seul le WER se comprend directement.
- Dans ce même test, on a observé un CER de **124 %** à l'itération 1 et un WER de **109,9 %** à l'itération 75. Ce sont des insertions : le modèle produisait plus de lettres, ou de mots, que la phrase n'en contenait.

**Code** (`stt/metrics.py`) :

```python
def edit_distance(ref, hyp):
    """Nombre minimum de substitutions + suppressions + insertions pour passer de ref à hyp."""
    previous_row = list(range(len(hyp) + 1))        # passer de "" à hyp[:j] = j insertions
    for i, r in enumerate(ref, start=1):
        row = [i]                                   # passer de ref[:i] à "" = i suppressions
        for j, h in enumerate(hyp, start=1):
            row.append(min(previous_row[j] + 1,                 # suppression
                           row[j - 1] + 1,                      # insertion
                           previous_row[j - 1] + (r != h)))     # substitution (0 si identique)
        previous_row = row
    return previous_row[-1]


def error_rate(references, hypotheses, unit):
    """WER (unit="mot") ou CER (unit="caractère"), en %."""
    split = (lambda s: s.split()) if unit == "mot" else list
    errors = sum(edit_distance(split(r), split(h)) for r, h in zip(references, hypotheses))
    total = sum(len(split(r)) for r in references)
    return 100 * errors / total
```

---

## 22. Le test d'overfitting (*sanity check*) et la boucle d'entraînement

**Explication.**
- **La boucle d'entraînement.** Une **itération** enchaîne quatre étapes :
  1. le **forward** (on calcule la sortie) ;
  2. la **loss** (on mesure l'erreur) ;
  3. la **backpropagation** (`loss.backward()` calcule, pour chaque poids, dans quel sens le bouger) ;
  4. le **pas de l'optimizer** (`optimizer.step()` bouge les poids).

  `optimizer.zero_grad()` efface les gradients de l'itération précédente, car PyTorch les **additionne** par défaut.
- **Le test d'overfitting.** Avant un long entraînement, on vérifie que le modèle est **capable d'apprendre par cœur** un tout petit jeu de données. S'il n'y arrive pas, il y a un bug : une forme fausse, une mauvaise loss ou des poids non entraînés. C'est une pratique standard de l'industrie, appelée *sanity check* (« overfit a single batch »).
- **`.train()` / `.eval()`.** Le dropout n'est actif qu'en mode `.train()`. En `.eval()`, il est coupé : les résultats sont stables et un peu meilleurs.

**Où on l'a appliquée.** `stt_03_overfit_10_clips.ipynb` : 10 clips, 400 itérations sur CPU.

**Pourquoi.**
- Un entraînement complet sur Kaggle dure des heures. Découvrir un bug au bout de 5 h coûterait cher. Ce test l'a vérifié en 23 min.
- **Résultat :**
  - la loss passe de 6,41 à environ 0,02 ;
  - en `.eval()`, on obtient un WER de **3,7 %** et un CER de **0,6 %**, et les phrases sont recopiées mot pour mot ;
  - **le pipeline fonctionne.**
- **Le dropout se voit dans les chiffres :**
  - en `.train()`, le WER oscillait entre 6 et 14 % sur le plateau ;
  - en `.eval()`, il tombe à 3,7 %.
- **Le piège du format :** `nn.CTCLoss` attend le temps en **premier**, `(T', B, C)`, alors que le modèle sort `(B, T', C)`. D'où le `transpose(0, 1)`.

**Code** (`stt_03_overfit_10_clips.ipynb`) :

```python
model.train()                                    # dropout actif (mode entraînement)
for step in range(1, N_STEPS + 1):
    # 1. Forward : log-probabilités (B, T', 50) et longueurs réduites
    log_probs, out_lengths = model(features, feature_lengths)
    # 2. Loss CTC : nn.CTCLoss attend le temps en PREMIER -> (T', B, 50)
    loss = ctc_loss(log_probs.transpose(0, 1), targets, out_lengths, target_lengths)
    # 3. Backpropagation : on efface les anciens gradients, puis on calcule les nouveaux
    optimizer.zero_grad()
    loss.backward()
    # Gradient clipping : plafonne la norme à 1 pour éviter un pas trop brutal
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    # 4. Mise à jour des poids
    optimizer.step()
```

---

## 23. Le scheduler : warmup puis décroissance cosinus

**Explication.** Le *scheduler* fait varier le learning rate au fil de l'entraînement. Le nôtre a deux phases :
- **Warmup :** le learning rate monte en ligne droite, de presque 0 jusqu'à son maximum. Au début, les poids sont aléatoires et les gradients sont grands et désordonnés. Un petit learning rate empêche ces premiers pas de « casser » le modèle.
- **Décroissance cosinus :** il redescend ensuite en demi-vague, `0,5 × (1 + cos(π × progression))`. La progression va de 0 à 1, donc le cosinus va de 1 à −1, et le multiplicateur de 1 à 0. La baisse est lente au début, rapide au milieu, puis de nouveau lente à la fin. À la fin, les pas sont minuscules : le modèle **affine** sa solution au lieu de sauter autour.

Exemple : avec un maximum de 1e‑3 et un warmup de 4 060 itérations, le learning rate vaut 1e‑3 × 1/4 060 = 2,5e‑7 à l'itération 0. Il vaut 1e‑3 à la fin du warmup, puis 5e‑4 au **milieu de la descente**. Ce milieu tombe à l'époque (10 + 200) / 2 = **105**, et non 100, parce que la descente ne commence qu'à la fin du warmup.

`LambdaLR` (PyTorch) multiplie le learning rate maximum par la valeur que renvoie **notre** fonction. On appelle `scheduler.step()` **à chaque itération**, juste après `optimizer.step()`.

**Où on l'a appliquée.** La fonction `warmup_cosine` a été testée dans la cellule 4 du notebook Kaggle. Elle sera utilisée par la boucle d'entraînement.

**Pourquoi.**
- **Le warmup.** Dans le test d'overfitting, la norme du gradient valait **18,7** à la 1re itération, contre 0,2 à 2 ensuite.
- **Le cosinus plutôt que Noam.** La durée totale est connue : 200 époques × 406 = 81 200 itérations. C'est aussi le standard actuel (wav2vec2, Whisper).
- **Le cosinus plutôt qu'un learning rate constant.** Un learning rate constant n'a ni warmup ni affinage.
- **Vérification :** les quatre valeurs calculées à l'avance ont été retrouvées avec le vrai `LambdaLR` (2,46e‑07 ; 1,00e‑03 ; 5,00e‑04 ; 4,15e‑13).

**Code :**

```python
def warmup_cosine(step, warmup_steps, total_steps):
    """Multiplicateur du learning rate (entre 0 et 1) à l'itération `step`."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps                              # montée linéaire : ~0 -> 1 (+1 : le 1er pas n'est pas nul)
    progress = (step - warmup_steps) / (total_steps - warmup_steps)  # 0 -> 1 pendant la décroissance
    return 0.5 * (1 + math.cos(math.pi * progress))                   # demi-vague cosinus : 1 -> 0

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: warmup_cosine(step, WARMUP_STEPS, TOTAL_STEPS))
```
