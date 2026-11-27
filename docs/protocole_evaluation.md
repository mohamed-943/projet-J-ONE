# Protocole d'évaluation du système vocal

Ce protocole mesure quatre indicateurs sur le robot, avec ses micros et sa configuration réelle :

| Indicateur | Ce qu'il mesure | Outil |
|---|---|---|
| WER | qualité de la transcription | `tools/evaluation.py` |
| Recognition Accuracy | reconnaissance des commandes adressées à Sarah | `tools/evaluation.py` |
| Latency | rapidité, de la fin de parole au premier son | journal des échanges + `tools/evaluation.py` |
| FAR / FRR | fiabilité de la détection vocale (parole et mot d'éveil) | `tools/evaluation.py` |

On évalue d'abord le mode local (LLM sur la Jetson, `--no-cloud`), puis le mode cloud.

## 1. Corpus

Le corpus suit `tests/protocole_evaluation.tsv`. Il comporte 56 enregistrements de quatre types :

| Type | Nombre | Contenu | Sert à |
|---|---|---|---|
| `phrase` | 22 (12 fr, 10 en) | questions et affirmations courantes | WER, FRR parole, langue |
| `commande` | 16 (8 fr, 8 en) | « Sarah, … » suivi d'une consigne, avec ses mots-clés | WER, Recognition Accuracy, FRR mot d'éveil |
| `hors_cible` | 10 (5 fr, 5 en) | parole entre personnes, sans « Sarah » | WER, FAR mot d'éveil |
| `bruit` | 8 | silence, pas, porte, clavier, applaudissements, chute d'objet, musique, robot debout | FAR parole, FAR mot d'éveil |

Conditions d'enregistrement, à noter dans le rapport :

- micros du robot (`--input g1_multicast`), locuteur face au robot, entre 0,5 et 1 m ;
- pièce et niveau de bruit ambiant ;
- nombre de locuteurs. Pour un résultat généralisable, faire enregistrer le corpus par au moins 3 personnes (un dossier par personne), puis évaluer chaque dossier.

## 2. Définitions

Chaque enregistrement passe par la même chaîne que dans le pipeline : découpage par le VAD (Silero), transcription (Whisper), filtre des transcriptions douteuses, puis mot d'éveil. Le mot d'éveil est évalué même s'il est désactivé en écoute continue.

**WER** (Word Error Rate), sur les enregistrements de parole :

WER = (S + D + I) / N

avec S les substitutions, D les suppressions, I les insertions et N le nombre de mots des références. Le texte est mis en minuscules, sans ponctuation, et les apostrophes séparent les mots. Une phrase que le VAD ne détecte pas compte en suppressions. Le WER est donné au global et par langue.

**Recognition Accuracy**, sur les 16 commandes :

Accuracy = commandes reconnues / commandes

Une commande est reconnue si (1) le mot d'éveil est détecté dans une transcription acceptée par le filtre et (2) tous ses mots-clés y figurent, sans tenir compte de la casse ni des accents. Le rapport donne aussi le taux de mots-clés présents dans la transcription brute : l'écart entre les deux taux montre ce que perdent le filtre et le mot d'éveil.

**Latency** :

- transcription : durée de Whisper par énoncé (p50, p90, max) ;
- bout en bout : de la fin de parole au premier son du robot, lue dans le journal des échanges (`latences_ms.bout_en_bout`) pour le LLM évalué. La cible ENF-01 est un p90 ≤ 4 000 ms en local et ≤ 2 500 ms en cloud.

**FAR / FRR**, à deux niveaux :

| Niveau | FAR (False Acceptance Rate) | FRR (False Rejection Rate) |
|---|---|---|
| Parole (VAD + filtre) | bruits qui auraient déclenché une réponse / bruits | enregistrements parlés sans aucun énoncé accepté / enregistrements parlés |
| Mot d'éveil | `hors_cible` et bruits acceptés par le mot d'éveil / (`hors_cible` + bruits) | commandes où « Sarah » n'est pas reconnu / commandes |

Chaque proportion est donnée avec son intervalle de confiance de Wilson à 95 %. Avec 8 bruits, un FAR de 0/8 laisse un intervalle de 0 à 32 % : il faut augmenter le nombre de bruits pour conclure finement.

## 3. Procédure (mode local)

Sur le robot, dans `~/projet-J-ONE` avec le venv activé :

1. **Enregistrer le corpus**, environ 15 min par personne :

   ```bash
   python tools/enregistrer_corpus.py --protocole tests/protocole_evaluation.tsv \
       --out corpus/eval_local/personne1 --input g1_multicast
   ```

   Pour chaque ligne : Entrée, lire la phrase, Entrée. Les bruits durent 6 s : Entrée, puis produire le bruit indiqué sans parler. Une séance interrompue reprend où elle s'était arrêtée.

2. **Mesurer la latence en conversation**. Faire au moins 20 échanges en mode local, en français et en anglais :

   ```bash
   python -m jone --no-cloud
   ```

3. **Évaluer** :

   ```bash
   python tools/evaluation.py --corpus corpus/eval_local/personne1 --llm local --depuis 2026-11-28T10:00
   ```

   `--depuis` limite la latence de bout en bout aux échanges de la séance de l'étape 2 : indiquer l'heure de son début.

   Le rapport est écrit dans `docs/evaluation_local-<date>.md`. Le détail par enregistrement est dans `logs/evaluation-<date>.json`.

## 4. Limites

- La Recognition Accuracy porte sur la reconnaissance de la consigne (mot d'éveil et mots-clés), pas sur la qualité de la réponse du LLM.
- Les bruits du protocole ne couvrent pas tous les environnements. Pour une démonstration dans une salle pleine, ajouter des enregistrements de conversations lointaines.
- La latence de bout en bout dépend du LLM et de la longueur des réponses : elle se mesure en conversation réelle, pas sur le corpus.
