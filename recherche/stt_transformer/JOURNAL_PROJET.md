# Journal de projet — STT / TTS pour le robot Unitree G1

**Projet de fin d'études :** Conception d'un système d'interaction homme-robot basé sur l'IA générative et la reconnaissance vocale.
**Responsabilité :** composants STT (Speech-to-Text) et TTS (Text-to-Speech), en français d'abord, puis en anglais.
**Méthode :** Comprendre → Explorer → Expérimenter → Comparer → Décider ensemble.

Ce journal recense chaque avancée, décision et résultat du projet, pour pouvoir rédiger le rapport final. Chaque entrée indique **ce qui a été fait**, **ce qu'on a observé** et **ce qu'on a décidé, avec la justification**.

---

## Environnement de travail

| Ressource | Caractéristiques | Usage |
|---|---|---|
| PC personnel | Windows 10, 4 cœurs, 16 Go de RAM, pas de GPU | Écriture et validation du code, prototypage sur de petits volumes |
| Kaggle Notebooks | GPU NVIDIA Tesla T4 (~15 Go de VRAM), 30 Go de RAM, 30 h de GPU par semaine, sessions de 12 h max, 20 Go conservés dans `/kaggle/working`, ~57 Go de disque par session | Entraînement des modèles |
| Machine GPU de l'équipe | GPU NVIDIA | Option pour les entraînements longs |
| Robot Unitree G1 | — | Intégration finale |

Outils : Python 3.12, gestionnaire de projet `uv`, PyTorch, NumPy, Matplotlib, pandas, VS Code et notebooks Jupyter.

---

## 2026-10-28 — Étape 1 : comprendre le signal audio

Notebook : `TTS.ipynb` (expériences pédagogiques, pas destinées au robot)

### Expérience 1 : forme d'onde d'un signal artificiel

- **Fait :** génération d'une sinusoïde à 440 Hz, échantillonnée à 16 kHz pendant 3 s.
- **Observé :** 48 000 échantillons pour 3 s. Un extrait de 200 échantillons dure 200 / 16 000 = 0,0125 s et contient 440 × 0,0125 = **5,5 cycles**, ce que confirme la courbe.
- **Retenu :**
  - Il ne faut pas confondre la **fréquence du signal** (440 Hz, nombre d'oscillations par seconde) et la **fréquence d'échantillonnage** (16 kHz, nombre de mesures par seconde).
  - D'après le théorème de Nyquist, un échantillonnage à 16 kHz représente correctement les fréquences jusqu'à **8 kHz**. Au-delà apparaît le phénomène d'**aliasing**. Cette bande suffit pour l'essentiel de la voix.

### Expérience 2 : la FFT

- **Fait :** FFT sur une fenêtre de 400 échantillons (25 ms), d'abord sur la sinusoïde à 440 Hz, puis sur un mélange 440 Hz + 1000 Hz (amplitudes 1 et 0,5).
- **Observé :** un pic unique à 440 Hz pour la sinusoïde seule. Pour le mélange, deux pics distincts à 440 Hz et 1000 Hz, de hauteurs 200 et 100, donc dans le même rapport que les amplitudes. La forme d'onde, elle, ne permettait pas de distinguer les deux fréquences.
- **Retenu :** la **résolution fréquentielle** vaut fréquence d'échantillonnage / taille de fenêtre, soit 16 000 / 400 = **40 Hz**. Il existe donc un **compromis temps / fréquence** : une grande fenêtre est précise en fréquence mais floue dans le temps, et inversement.

### Expérience 3 : le STFT et le spectrogramme

- **Fait :** écriture à la main d'un STFT (fenêtres de 25 ms décalées de 10 ms, une FFT par fenêtre), puis comparaison de trois tailles de fenêtre (100, 400 et 1 600 échantillons). Le notebook est prêt, mais les résultats n'ont pas été relevés : la priorité a été donnée au démarrage du projet.
- **Retenu :** le spectrogramme est une matrice **(temps × fréquences)**. C'est le type d'entrée que recevra notre modèle. La représentation **log-Mel** sera étudiée pendant le prétraitement.

---

## 2026-10-28 — Décisions de cadrage

| Décision | Choix | Justification |
|---|---|---|
| Approche | **Concevoir et entraîner nos propres Transformers**, sans utiliser de modèles STT ou TTS tout faits | C'est l'objectif du projet : maîtriser tout le cycle (forward pass, loss, backpropagation, optimisation) |
| Ordre | **STT d'abord**, puis TTS | Le STT est l'entrée de la chaîne homme-robot, et son évaluation (WER) est objective |
| Portée du STT | **Progressive** : d'abord faire fonctionner toute la chaîne sur un petit volume de français libre, puis décider d'élargir (plus de données) ou de spécialiser (commandes du robot) | Valider le pipeline avant d'investir des heures de GPU |
| Accélérateur | **GPU T4** plutôt que TPU | PyTorch standard avec CUDA. Le TPU demande la bibliothèque `torch_xla` et gère mal les entrées de longueur variable, ce qui est le cas de l'audio |
| Plateforme d'entraînement | **Kaggle** (30 h de GPU par semaine) | Gratuit, avec assez de disque pour le dataset complet pendant une session |

---

## 2026-10-28 — Choix et acquisition des données

**Source :** Mozilla Common Voice, distribué depuis octobre 2025 exclusivement par Mozilla Data Collective (MDC). Licence CC0 avec conditions d'utilisation MDC : **interdiction de réhéberger ou de repartager les données**, et interdiction de chercher à identifier les locuteurs.

| Dataset | Taille | Clips | Rôle |
|---|---|---|---|
| CV Scripted Speech 26.0 — French of France (segment) | 167 Mo | 5 490 | Entraînement |
| CV Scripted Speech 26.0 — Canadian French (segment) | 291 Mo | 8 448 | Entraînement (plus d'heures et d'accents variés) |
| CV Spontaneous Speech 5.0 — French | 48,5 Mo | 716 mp3, 688 lignes | Parole naturelle : une partie pour l'entraînement, une partie pour le test, découpé par locuteur |
| CV Scripted Speech 27.0 — French (complet) | 28,55 Go | ~1 200 h | Plus tard, pour l'entraînement réel sur Kaggle |

**Décisions :**
- Les données sont stockées dans `data/`, **exclu de git** à cause de la licence et de la taille.
- **Découpage en trois parts : train / validation / test.**
  - La validation sert aux réglages pendant le développement.
  - Le test n'est utilisé qu'une fois, à la fin, pour le score final.
  - On utilise le découpage officiel de Common Voice (`train.tsv`, `dev.tsv`, `test.tsv`).
- **Découpage par locuteur :** une même voix ne doit jamais apparaître dans deux parts différentes. Sinon le WER mesuré serait trop optimiste, car le modèle aurait déjà entendu ces voix pendant l'entraînement : c'est une **fuite de données**.
- Le dataset spontané sera découpé par locuteur entre entraînement et test, selon sa taille réelle.

---

## 2026-10-29 — EDA (analyse exploratoire des données)

Notebook : `stt_01_exploration.ipynb`

Plan : chaque point doit aboutir à une décision pour le modèle.

1. Transcriptions manquantes
2. Inventaire des caractères (vocabulaire CTC, normalisation du texte)
3. Durée des clips (mémoire GPU, padding)
4. Fréquence d'échantillonnage des mp3 (rééchantillonnage)
5. Rapport longueur du texte / durée de l'audio (contrainte du CTC)
6. Répartition des locuteurs
7. Dataset spontané : découpage et qualité

### Point 1 : transcriptions manquantes — fr-FR train

- **Observé :** 5 086 clips et 13 colonnes. **`sentence` : 0 valeur manquante.**
- `age` (642 manquants), `gender` (1 732) et `variant` (899) sont souvent vides : ces champs sont facultatifs et remplis par les contributeurs eux-mêmes.
- `accents` est toujours rempli, parce que le segment a été construit à partir de cette information.
- `sentence_domain` et `segment` sont presque entièrement vides.
- **Décision :** pour l'entraînement, on ne garde que `path` (le fichier audio), `sentence` (la cible) et `client_id` (le locuteur).

### Point 1 (suite) : fr-CA et dataset spontané

- **fr-CA train :** 8 077 clips, **0 phrase manquante**.
- **Spontané :** 688 lignes, dont **394 sans transcription (57 %)**. Une réponse spontanée doit être transcrite par un humain après coup, et beaucoup ne l'ont pas encore été. La colonne `split` est entièrement vide : Mozilla ne fournit aucun découpage officiel.
- **Données spontanées utilisables :** **294 clips**, **41 locuteurs**, **65 minutes** au total.
- **Répartition très inégale :** le locuteur le plus actif a 45 clips, la moitié des locuteurs en a 3 ou moins, et un quart n'en a qu'un seul.
- **Décision :** on ne conserve que les clips transcrits, et **tout le spontané sert de jeu de test** (option A). Avec 65 minutes, l'ajouter aux ~19 h d'entraînement n'aurait que peu d'effet. En revanche, garder les 294 clips et les 41 locuteurs en test donne une mesure plus fiable du WER sur de la parole naturelle, proche de ce qu'entendra le robot. Un découpage 70 / 30 aurait laissé environ 90 clips en test, ce qui rend le WER instable.
- À vérifier au point 7 : l'écart entre 716 mp3 et 688 lignes.

### Point 2 : inventaire des caractères et normalisation du texte

- **Observé (train fr-FR + fr-CA, 13 163 phrases) :** **178 caractères différents**.
  - Les majuscules sont comptées à part.
  - Ponctuation variée.
  - Deux apostrophes coexistent : `'` et `’`.
  - Certains accents sont « décomposés » : `e` suivi d'un accent isolé, au lieu d'un seul caractère `é`.
  - On trouve des lettres étrangères (ñ, š, ł…), du grec et des symboles (♫, ∞), souvent vus une seule fois.
  - **Aucun chiffre** : Common Voice fait écrire les nombres en lettres.
- **Problème :** un caractère vu une seule fois ne peut pas être appris, mais il occupe quand même une case dans la sortie du modèle. Majuscules et ponctuation ne s'entendent pas, ou très peu.
- **Décisions de normalisation :**
  - normalisation Unicode **NFC**, qui recolle les accents décomposés ;
  - **tout en minuscules** ;
  - toutes les apostrophes sont unifiées en `'` ;
  - **ponctuation gardée : `. , ? !`**, plus `'` et `-`, qui font partie des mots (« l'homme », « peut-être ») ;
  - la ponctuation secondaire (`; : … « » ( ) —`) est remplacée par **un espace**. Si on la supprimait, deux mots pourraient se retrouver collés (« bonjour;merci » deviendrait « bonjourmerci »). Les espaces en trop sont ensuite réduits à un seul ;
  - les phrases contenant un caractère hors de l'alphabet français sont **écartées**.
- **Résultat :** **181 phrases écartées sur 13 163 (1,4 %)**. Il reste **49 caractères** : 26 lettres, 16 lettres accentuées ou ligatures françaises, l'espace et 6 signes de ponctuation.
- **Conséquence pour le modèle :** l'output head produira **50 scores** à chaque instant : les 49 caractères, plus le **blank** du CTC.

### Choix de la bibliothèque audio

- **Test :** les versions récentes de torchaudio délèguent la lecture des fichiers à TorchCodec, qui exige FFmpeg en version « shared ». Sous Windows, la lecture des mp3 a échoué avec l'installation de FFmpeg présente sur le PC. La bibliothèque **soundfile** (libsndfile 1.2.2) lit en revanche les mp3 sans problème.
- **Décision :** **soundfile pour lire les fichiers audio**, et **torchaudio pour les traitements** (rééchantillonnage, log-Mel). Le même code fonctionne sous Windows et sous Linux (Kaggle), sans installation système.

### Points 3 et 4 : durée des clips et fréquence d'échantillonnage

| | fr-FR | fr-CA | Total |
|---|---|---|---|
| Durée totale | 7,12 h | 11,81 h | **18,93 h** |
| Durée médiane | 4,9 s | 5,2 s | |
| Minimum / maximum | 1,7 s / 10,6 s | 1,5 s / 11,2 s | |
| Fréquences d'échantillonnage | 32 kHz : 3 911 · 48 kHz : 1 161 · 44,1 kHz : 14 | 48 kHz : 4 413 · 32 kHz : 3 664 | **3 fréquences différentes** |

- **Observé :** chaque clip correspond à une seule phrase lue, d'où des durées courtes et homogènes. Les fréquences d'échantillonnage sont hétérogènes : une même seconde de parole ne donne pas le même nombre d'échantillons selon le clip.
- **Décision 1 :** tout **rééchantillonner à 16 000 Hz**.
  - C'est la norme en reconnaissance vocale, ce qui rend nos résultats comparables à ceux de la littérature.
  - D'après le théorème de Nyquist, on conserve les fréquences jusqu'à 8 kHz, ce qui suffit pour la parole.
  - Les données sont 2 à 3 fois plus légères qu'en 32 ou 48 kHz : un clip de 5 s compte 80 000 échantillons au lieu de 240 000.
- **Décision 2 :** **garder tous les clips**. Les durées sont homogènes, et aucun clip ne pose de problème de mémoire. La contrainte du CTC sur les clips courts sera vérifiée au point 5.

### Point 5 : la contrainte du CTC

- **Principe :** le CTC exige que le nombre d'instants de sortie soit **supérieur ou égal** au nombre de caractères de la phrase, plus le nombre de lettres doublées (un blank est obligatoire entre deux lettres identiques). On mesure donc combien d'instants par seconde nos phrases demandent, pour savoir de combien le modèle pourra réduire le nombre d'instants (sous-échantillonnage temporel). Cette réduction allège l'attention, dont le coût croît avec le carré du nombre de positions.

| Instants nécessaires par seconde | fr-FR | fr-CA |
|---|---|---|
| Médiane | 12,7 | 11,8 |
| Maximum | 21,8 | 22,6 |
| Clips impossibles avec ÷ 4 (25 instants/s) | **0** | **0** |
| Clips impossibles avec ÷ 8 (12,5 instants/s) | **2 707 (53 %)** | **3 136 (39 %)** |

- **Conclusion** (avec une analyse à 10 ms, soit 100 instants par seconde) :
  - une réduction **÷ 8 est exclue**, car elle rendrait impossible la transcription de 44 % des clips ;
  - une réduction **÷ 4 est possible**, mais la marge est faible : 25 instants/s disponibles, pour un maximum observé de 22,6 ;
  - une réduction ÷ 2, ou aucune réduction, laisse une marge confortable, mais coûte plus cher en calcul.
- **Décision reportée** à la conception de l'architecture : le choix se fera entre ÷ 2 et ÷ 4.
- **À vérifier :** un clip ne demande que 1,05 instant par seconde.

### Point 6 : répartition des locuteurs

| | fr-FR | fr-CA |
|---|---|---|
| Locuteurs différents | 106 | 169 |
| Clips par locuteur (médiane) | 14,5 | 18 |
| Locuteur le plus actif | 505 clips (≈ 10 % du dataset) | 909 clips (≈ 11 %) |
| Part des 10 locuteurs les plus actifs | **53,8 %** | **43,6 %** |

- **Observé :** il y a peu de voix différentes (275 locuteurs en entraînement, contre environ 21 000 dans le dataset complet), et leur répartition est très déséquilibrée. C'est typique de Common Voice, où quelques bénévoles très actifs enregistrent la majorité des phrases.
- **Risque :** le modèle risque de s'adapter aux voix dominantes, ce qui serait une forme d'overfitting sur les locuteurs. Les jeux de dev et de test, composés d'autres locuteurs, permettront de le détecter.
- **Options étudiées :**
  - A. Tout garder.
  - B. Plafonner à 200 clips par locuteur : −17,7 % de clips.
  - C. Plafonner à 100 clips par locuteur : −35,2 % de clips.
  - D. Échantillonnage pondéré pendant l'entraînement.
- **Décision : A, tout garder pour le prototype.** Avec 19 h de données, chaque clip compte, et le prototype sert d'abord à valider la chaîne complète. Le déséquilibre sera beaucoup plus faible avec le dataset complet (MDC, 28,5 Go). Si l'écart de WER entre train et dev est important, l'option D sera la première piste à essayer.

### Point 7 : dataset spontané (fichiers, qualité, conventions de transcription)

- **716 mp3 pour 688 lignes :** les 28 mp3 en trop sont exactement les 28 enregistrements **signalés** par la communauté (`ss-reported-audios-fr.tsv`) : 27 pour une réponse dans une autre langue, 1 pour une autre raison. Ils sont déjà exclus du tableau principal.
- **Étiquettes de qualité (294 clips transcrits) :** 49 clips sont marqués `transcription-length` (transcription anormalement courte pour la durée de l'audio, donc probablement incomplète), dont 2 également `short-audio`.
- **Conventions de transcription différentes de la parole lue :**
  - annotations entre crochets dans 44 transcriptions : `[disfluency]`, `[noise]`, `[toux]`, `[no sound]`, `[unclear]`, `[pas clair]` ;
  - chiffres dans 6 transcriptions ;
  - hésitations « euh » dans 26 transcriptions.
  - Notre normalisation en écartait 54 sur 294.
- **Durées beaucoup plus longues** que pour l'entraînement : médiane de 8,9 s, maximum de **83 s**, et 25 % des clips au-delà de 17 s. Les clips d'entraînement ne dépassent pas 11,2 s. Or un Transformer généralise souvent mal à des longueurs jamais vues.

**Règles d'évaluation décidées pour le jeu de test spontané :**

| Règle | Justification |
|---|---|
| Écarter les 49 clips `transcription-length` | Une référence incomplète pénaliserait le modèle pour des mots réellement prononcés |
| Écarter les clips contenant `[unclear]` ou `[pas clair]` | Passage incompréhensible, donc référence incomplète |
| Écarter les clips contenant des chiffres | L'entraînement n'en contient aucun : le modèle ne peut pas en produire |
| Supprimer `[noise]`, `[toux]`, `[disfluency]` et `[no sound]` des références | Ce ne sont pas des mots. Le spontané sert uniquement au test : le modèle n'apprendra jamais ces étiquettes, qui ne feraient que gonfler le WER |
| Retirer les « euh » des références | Pratique courante en évaluation STT : on mesure les mots porteurs de sens. Un modèle entraîné sur de la parole lue ne connaît pas « euh » |
| Garder les clips longs, et calculer le WER **par tranche de durée** (moins de 11 s, de 11 à 30 s, plus de 30 s) | Mesurer et documenter l'effet de la longueur, plutôt que de le cacher |

**Jeu de test spontané final :** **230 clips**, **38 locuteurs**, **50,8 minutes**. Répartition par durée : 131 clips de moins de 11 s, 79 de 11 à 30 s, et 20 de plus de 30 s.

**Piste d'amélioration future :** détecter les hésitations, les bruits et la toux, ce qui serait utile pour l'interaction avec le robot. Cela demanderait des données d'**entraînement** annotées pour ces événements, par exemple nos propres enregistrements avec le robot. C'est peut-être un long silence, ou un texte qui ne correspond pas à l'audio. Hypothèse : il s'agit d'enregistrements signalés par la communauté (`ss-reported-audios-fr.tsv`).
