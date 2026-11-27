from pathlib import Path

from jone.evaluation import Item, ItemResult, compute_metrics, keywords_found, load_protocol, wilson

ROOT = Path(__file__).resolve().parent.parent


def test_protocol_file_is_valid():
    items = load_protocol(ROOT / "tests" / "protocole_evaluation.tsv")
    types = {i.type for i in items}
    assert types == {"phrase", "commande", "hors_cible", "bruit"}
    assert len({i.id for i in items}) == len(items)
    commands = [i for i in items if i.type == "commande"]
    assert all(i.texte.lower().startswith("sibo") and i.mots_cles for i in commands)
    assert {i.langue for i in items if i.is_speech} == {"fr", "en"}


def test_keywords_ignore_case_and_accents():
    assert keywords_found("Sibo, qui t'a CRÉÉ ?", ["cré"])
    assert keywords_found("Sibo, repete s'il te plait", ["répète"])
    assert not keywords_found("Sibo, parle plus vite", ["fort"])


def test_wilson_interval():
    p, lo, hi = wilson(9, 10)
    assert p == 0.9 and 0.59 < lo < 0.6 and 0.98 < hi < 0.99
    assert wilson(0, 0) == (None, None, None)


def test_metrics_far_frr_and_accuracy():
    phrase = ItemResult(Item("p1", "phrase", "fr", "bonjour à tous"), transcription="bonjour a tous",
                        accepted="bonjour a tous", language="fr")
    cmd_ok = ItemResult(Item("c1", "commande", "fr", "Sibo, parle plus fort.", ["fort"]),
                        transcription="Sibo, parle plus fort.", accepted="Sibo, parle plus fort.",
                        wake=True, language="fr")
    cmd_missed = ItemResult(Item("c2", "commande", "en", "Sibo, stop talking.", ["stop"]),
                            transcription="Sara stop talking", accepted="", language="en")
    off_target = ItemResult(Item("h1", "hors_cible", "fr", "on mange où"), transcription="on mange où",
                            accepted="on mange où", language="fr")
    noise_ok = ItemResult(Item("b1", "bruit", "-", "silence"))
    noise_bad = ItemResult(Item("b2", "bruit", "-", "porte"), transcription="Merci.", accepted="Merci.")
    m = compute_metrics([phrase, cmd_ok, cmd_missed, off_target, noise_ok, noise_bad])

    assert m["reconnaissance_commandes"]["k"] == 1 and m["reconnaissance_commandes"]["n"] == 2
    assert m["mots_cles_seuls"]["k"] == 2  # « stop » est bien transcrit, mais rejeté ensuite
    assert m["parole"]["FRR"]["k"] == 1 and m["parole"]["FRR"]["n"] == 4  # c2 perdu
    assert m["parole"]["FAR"]["k"] == 1 and m["parole"]["FAR"]["n"] == 2  # la porte
    assert m["mot_eveil"]["FRR"]["k"] == 1 and m["mot_eveil"]["FAR"]["k"] == 0
    assert m["identification_langue"]["taux"] == 1.0
    assert 0 < m["wer"]["global"] < 0.5 and "en" in m["wer"]


def test_audio_levels_and_snr():
    import numpy as np

    from jone.evaluation import audio_levels

    rng = np.random.default_rng(0)
    noise = rng.normal(0, 30, 16000)
    voice = rng.normal(0, 3000, 16000)  # 40 dB au-dessus
    v, n = audio_levels(np.concatenate([noise, voice]).astype(np.int16))
    assert 37 < v - n < 43


def test_distance_folders_and_max_distance(tmp_path):
    from evaluation_distance import distance_dirs, max_distance

    for name in ("3m", "0.5m", "150cm", "1m", "notes"):
        (tmp_path / name).mkdir()
    assert [d for d, _ in distance_dirs(tmp_path)] == [0.5, 1.0, 1.5, 3.0]

    rows = [{"distance_m": 0.5, "wer": 0.02, "frr": 0.0}, {"distance_m": 1.0, "wer": 0.05, "frr": 0.0},
            {"distance_m": 2.0, "wer": 0.18, "frr": 0.1}, {"distance_m": 3.0, "wer": 0.06, "frr": 0.0}]
    # 3 m passe, mais 2 m échoue : la distance retenue s'arrête à 1 m
    assert max_distance(rows, 0.10, 0.10) == 1.0


def test_confidence_curve():
    from jone.evaluation import confidence_curve

    good = ItemResult(Item("p1", "phrase", "fr", "bonjour"), segment_scores=[(True, -0.3)])
    weak = ItemResult(Item("p2", "phrase", "fr", "salut"), segment_scores=[(True, -1.2)])
    noise = ItemResult(Item("b1", "bruit", "-", "porte"), segment_scores=[(True, -1.5)])
    repeated = ItemResult(Item("b2", "bruit", "-", "clavier"), segment_scores=[(False, -0.2)])
    curve = {c["seuil"]: c for c in confidence_curve([good, weak, noise, repeated], (-1.6, -1.0))}
    assert curve[-1.6]["FRR"]["k"] == 0 and curve[-1.6]["FAR"]["k"] == 1
    assert curve[-1.0]["FRR"]["k"] == 1 and curve[-1.0]["FAR"]["k"] == 0  # « repeated » rejeté par une autre règle
