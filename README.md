# J-ONE — interaction vocale homme-robot

Pipeline vocal en français pour le robot humanoïde **J-ONE** (Unitree G1-EDU) : l'utilisateur parle, le robot comprend et répond à voix haute, sans clavier ni écran.

```
micro G1 ─► Silero VAD ─► Whisper turbo ─► filtre ─► routeur LLM ─► Piper TTS ─► PlayStream() ─► haut-parleur
 (multicast)  (parole ?)    (texte, local)  (EF-03)   │  Claude Haiku 5.5 (cloud)       (PCM 16 kHz)     │
                                                      └► Qwen3 4B (local, secours)                       │
     ▲                                                                                                    │
     └────────────────────────── réécoute après la fin réelle du son (EF-10) ◄──────────────────────────────┘
                         LED de la tête : vert écoute · bleu réflexion · blanc parole · rouge erreur
```

Tout tourne sur la Jetson Orin, sauf le LLM principal : seul le texte transcrit part vers l'API Anthropic, jamais l'audio (ENF-07). Sans Internet, le pipeline complet fonctionne avec le LLM local (ENF-03).

Le module vocal ne commande aucun mouvement : il n'utilise que le client audio du SDK (son et LED), jamais les clients de locomotion. Il ne peut donc pas interférer avec le contrôle moteur ni avec l'arrêt d'urgence.

## Arborescence

```
jone/
  config.py         valeurs par défaut + lecture de config.yaml et .env
  audio_input.py    micro du G1 (UDP multicast) ou du PC (sounddevice)
  vad.py            Silero VAD + découpage en énoncés (EF-01)
  stt.py            Whisper large-v3-turbo via faster-whisper (EF-02)
  text_filter.py    rejet des transcriptions parasites (EF-03) + mot d'éveil optionnel
  conversation.py   historique partagé par les deux LLM (EF-05)
  llm/claude.py     Claude Haiku 5.5 en streaming (EF-04)
  llm/local.py      Qwen3 4B via Ollama (secours)
  llm/router.py     bascule automatique cloud → local (EF-07, ENF-04)
  text_utils.py     découpage en phrases pour le streaming, nettoyage pour l'oral (EF-06)
  tts.py            Piper, voix fr_FR-siwis (EF-08)
  audio_utils.py    conversion en PCM 16 kHz mono 16 bits (EF-09)
  audio_output.py   PlayStream() sur le G1, ou haut-parleur du PC (EF-10)
  led.py            LedControl() selon l'état (EF-11)
  exchange_log.py   journal JSONL de chaque échange (EF-12)
  metrics.py        WER et statistiques de latence
  pipeline.py       la boucle écoute → réponse → réécoute
prompts/system_prompt_fr.txt   personnalité et règles de J-ONE
config.yaml                    configuration du robot
config.pc.yaml                 variante pour développer sur PC
tools/                         scripts de test et de mesure (phases 1, 2, 3, 5)
tests/                         tests unitaires (pytest)
docs/plan_de_tests.md          procédure de validation
docs/README_technique.md       fonctionnement interne, pour les développeurs
```

Pour le fonctionnement interne (fils d'exécution, interfaces des modules, format du journal, extension), voir la [documentation technique](docs/README_technique.md).

## Installation sur la Jetson Orin du G1-EDU

Prérequis : JetPack 6 (Python 3.10 ou plus), accès SSH à la Jetson (réseau interne `192.168.123.x`), robot en mode développeur pour les tests.

```bash
git clone <ce dépôt> projet-J-ONE && cd projet-J-ONE
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# SDK Unitree (PlayStream, LedControl)
git clone https://github.com/unitreerobotics/unitree_sdk2_python
pip install -e unitree_sdk2_python

# LLM local
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:4b

# Voix Piper
python tools/download_models.py

# Clé API Anthropic
cp .env.example .env   # puis renseigner ANTHROPIC_API_KEY
```

**Whisper sur le GPU de la Jetson.** Les paquets `ctranslate2` publiés pour aarch64 sont compilés sans CUDA : faster-whisper tournerait alors sur le CPU (le code le détecte et bascule en `int8` CPU, avec un avertissement dans le journal). Pour atteindre la latence visée, il faut l'une de ces deux solutions :
- compiler CTranslate2 avec CUDA sur la Jetson ;
- utiliser l'image `faster-whisper` de [jetson-containers](https://github.com/dusty-nv/jetson-containers), qui l'embarque déjà.

Vérifier au démarrage la ligne `Whisper large-v3-turbo chargé` et l'absence d'avertissement « repli sur CPU ».

## Lancement

```bash
python -m jone                      # robot : micros G1, PlayStream, LED
python -m jone --no-cloud           # forcer le LLM local (test hors ligne)
python -m jone --config config.pc.yaml           # PC : micro et haut-parleur de l'ordinateur
python -m jone --config config.pc.yaml --text    # PC, au clavier : teste LLM, mémoire, TTS
```

Options : `--input`, `--output`, `--led` surchargent la configuration (`--output none --led console` pour tester sans matériel). En mode texte, `/reset` efface la mémoire de conversation. `Ctrl+C` arrête proprement (LED éteinte).

## Configuration

Tous les réglages sont dans [config.yaml](config.yaml), commentés et rattachés à leur exigence. Les plus utiles à ajuster après les premières mesures :

| Clé | Effet |
|---|---|
| `vad.min_silence_ms` | silence qui marque la fin de phrase : plus bas = plus réactif, mais risque de couper l'utilisateur |
| `vad.threshold` | sensibilité à la parole ; à monter dans une salle bruyante |
| `stt.beam_size` | 1 = rapide ; 5 = un peu plus précis, plus lent |
| `llm.cloud.first_token_timeout_s` | seuil de bascule vers le LLM local (ENF-04) |
| `llm.cloud.retry_cooldown_s` | délai avant de retenter le cloud après une panne |
| `llm.cloud.provider` | `claude` (par défaut) ; `groq` pour essayer sans clé Anthropic (`GROQ_API_KEY` dans `.env`, modèle dans `llm.cloud.groq`) |
| `llm.history_max_turns` | nombre d'échanges gardés en mémoire |
| `wake_word.enabled` | `true` : le robot ne répond que si on dit « J-ONE » |
| `audio_output.tail_margin_s` | pause entre la fin du son et la reprise de l'écoute |

La personnalité de J-ONE se modifie dans [prompts/system_prompt_fr.txt](prompts/system_prompt_fr.txt), sans toucher au code.

## Fonctionnement détaillé

**Écoute (EF-01).** Le micro est lu par trames de 32 ms. Silero VAD donne une probabilité de parole par trame. Un énoncé commence au-dessus de `threshold` et se termine après `min_silence_ms` sous `neg_threshold`. Les énoncés de moins de `min_speech_ms` (porte, pas, choc) sont jetés, et 300 ms d'audio avant le déclenchement sont conservés pour ne pas perdre la première syllabe.

**Transcription et filtre (EF-02, EF-03).** Whisper transcrit en local. Le filtre rejette les textes vides, les hallucinations connues de Whisper (« Sous-titres réalisés par la communauté d'Amara.org »…), les textes répétitifs et ceux dont la confiance est trop basse. Les rejets sont journalisés avec leur motif.

**Réponse (EF-04 à EF-07).** Le routeur interroge Claude Haiku 5.5 en streaming, avec le prompt système et l'historique. Si aucun fragment n'arrive avant `first_token_timeout_s`, ou si la connexion échoue, il bascule aussitôt sur Qwen3 4B local et met le cloud de côté `retry_cooldown_s` secondes. Si la panne survient en cours de réponse, le robot s'en tient à ce qu'il a déjà dit. La réponse est coupée après 3 phrases quoi qu'il arrive, et nettoyée de tout ce qui ne se prononce pas.

**Synthèse et lecture (EF-08 à EF-10).** Dès qu'une phrase complète arrive du LLM, Piper la synthétise et elle part au haut-parleur, pendant que le LLM écrit la suite : c'est ce qui réduit la latence perçue. L'audio est converti en PCM 16 kHz mono 16 bits et envoyé à `PlayStream()` par blocs d'une seconde. Le programme connaît ainsi l'heure de fin réelle du son. Le micro est suspendu dès la fin de la phrase de l'utilisateur et jusqu'à `tail_margin_s` après la fin du son, puis ses tampons sont vidés : le robot ne s'entend pas lui-même.

**État et journal (EF-11, EF-12).** La LED suit chaque étape. Chaque énoncé produit une ligne dans `logs/echanges-AAAAMMJJ.jsonl` : horodatage, transcription, réponse, LLM utilisé, bascule, erreurs, latences (STT, premier jeton, première synthèse, bout en bout). `logs/jone.log` contient le journal technique.

**Robustesse (ENF-06).** Une exception pendant un tour coupe le son, allume la LED en rouge deux secondes, est journalisée, puis l'écoute reprend.

## Tests et mesures

```bash
pip install -r requirements-dev.txt && python -m pytest     # tests unitaires, sans matériel
python tools/test_g1_audio.py --say "Bonjour"                # phase 1 : PlayStream, LedControl, micro
python tools/bench_stt.py --corpus corpus/calme              # phase 2 : WER et latence Whisper
python tools/bench_tts.py                                     # phase 2 : latence Piper
python tools/bench_llm.py [--simulate-outage]                 # phase 3 : premier jeton, bascule
python tools/rapport_tests.py --out docs/rapport_mesures.md  # phase 5 : rapport depuis les journaux
```

La procédure complète de validation, critère par critère, est dans [docs/plan_de_tests.md](docs/plan_de_tests.md).

## Correspondance avec le cahier des charges

| Exigence | Où | Vérification |
|---|---|---|
| EF-01 détection de parole | `vad.py` | `tests/test_vad.py` |
| EF-02 STT local | `stt.py` | `tools/bench_stt.py` |
| EF-03 filtre | `text_filter.py` | `tests/test_text_filter.py` |
| EF-04 personnalité | `prompts/system_prompt_fr.txt`, `llm/claude.py` | conversation de démonstration |
| EF-05 historique | `conversation.py` | `tests/test_pipeline.py`, critère C1 |
| EF-06 1 à 3 phrases, sans symbole | prompt + `text_utils.py` + coupure dans `pipeline.py` | `tests/test_pipeline.py` |
| EF-07 bascule | `llm/router.py` | `tests/test_router.py`, `bench_llm.py --simulate-outage`, critère C3 |
| EF-08 voix française | `tts.py` | `tools/bench_tts.py` |
| EF-09 PCM 16 kHz mono 16 bits | `audio_utils.py` | `tests/test_misc.py` |
| EF-10 écoute suspendue | `pipeline.py`, `audio_input.py`, `audio_output.py` | critère C5 |
| EF-11 LED | `led.py` | `tools/test_g1_audio.py`, critère C4 |
| EF-12 journal | `exchange_log.py` | `tools/rapport_tests.py` |
| ENF-01 latence | streaming phrase par phrase | rapport, critère C2 |
| ENF-03 hors ligne | `llm/local.py` | critère C3 |
| ENF-04 seuil de bascule | `llm.cloud.first_token_timeout_s` | `tests/test_router.py` |
| ENF-07 confidentialité | seul le texte part au cloud | lecture de `llm/claude.py` |
| ENF-08 modules découplés | un module par étape, `config.yaml` | — |

## Points à vérifier sur le robot

- **Micro du G1.** Le code écoute le flux multicast `239.168.123.161:5555` (PCM 16 kHz mono 16 bits), depuis l'IP de la Jetson `192.168.123.164`. Ces valeurs viennent de la documentation Unitree pour le G1 et sont à confirmer avec `tools/test_g1_audio.py`, de même que le nombre de micros (le cahier des charges note l'écart entre la légende du pipeline et le manuel). En cas de doute, `audio_input.backend: sounddevice` permet d'utiliser un micro USB branché à la Jetson.
- **Format de `PlayStream()`.** Le code envoie des octets PCM bruts par blocs d'une seconde, avec un identifiant de flux par réponse. Vérifier sur le robot qu'il n'y a ni coupure entre les blocs ni accélération.
- **Interface réseau.** `robot.network_interface` (par défaut `eth0`) doit être celle du réseau `192.168.123.x`.

## Questions ouvertes du cahier des charges

- **Mot d'éveil ou écoute continue.** Les deux sont implémentés : écoute continue par défaut, mot d'éveil avec `wake_word.enabled: true`. Après un échange, le robot reste attentif `session_timeout_s` secondes sans qu'on répète « J-ONE ». La décision reste à prendre après les essais en salle.
- **Clé API et budget.** À décider par l'équipe. Avec Claude Haiku 5.5 (0,10 $ par million de jetons en entrée, 0,50 $ en sortie) et des échanges d'environ 1 500 jetons en entrée et 80 en sortie, une conversation de 100 échanges coûte de l'ordre de 0,02 $. Le poste de dépense reste négligeable même avec des démonstrations quotidiennes.

## Sécurité d'exploitation

Rappel des règles du manuel G1 : au moins 2 m entre le robot et les personnes, arrêt d'urgence toujours accessible (L2+B maintenu plus de 5 s), robot suspendu sur son cadre en mode développeur, sessions de moins de 2 h avec vérification de la batterie, utilisation intérieure entre 0 et 40 °C.
