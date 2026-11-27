from pathlib import Path

from jone.evaluation import Item, ItemResult, compute_metrics, keywords_found, load_protocol, wilson

ROOT = Path(__file__).resolve().parent.parent


def test_protocol_file_is_valid():
    items = load_protocol(ROOT / "tests" / "protocole_evaluation.tsv")
    types = {i.type for i in items}
    assert types == {"phrase", "commande", "hors_cible", "bruit"}
    assert len({i.id for i in items}) == len(items)
    commands = [i for i in items if i.type == "commande"]
    assert all(i.texte.lower().startswith("sarah") and i.mots_cles for i in commands)
    assert {i.langue for i in items if i.is_speech} == {"fr", "en"}


def test_keywords_ignore_case_and_accents():
    assert keywords_found("Sarah, qui t'a CRÉÉE ?", ["cré"])
    assert keywords_found("Sarah, repete s'il te plait", ["répète"])
    assert not keywords_found("Sarah, parle plus vite", ["fort"])


def test_wilson_interval():
    p, lo, hi = wilson(9, 10)
    assert p == 0.9 and 0.59 < lo < 0.6 and 0.98 < hi < 0.99
    assert wilson(0, 0) == (None, None, None)


def test_metrics_far_frr_and_accuracy():
    phrase = ItemResult(Item("p1", "phrase", "fr", "bonjour à tous"), transcription="bonjour a tous",
                        accepted="bonjour a tous", language="fr")
    cmd_ok = ItemResult(Item("c1", "commande", "fr", "Sarah, parle plus fort.", ["fort"]),
                        transcription="Sarah, parle plus fort.", accepted="Sarah, parle plus fort.",
                        wake=True, language="fr")
    cmd_missed = ItemResult(Item("c2", "commande", "en", "Sarah, stop talking.", ["stop"]),
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
