from jone.text_utils import SentenceSplitter, ThinkStripper, clean_for_speech


def feed_all(splitter, chunks):
    out = []
    for c in chunks:
        out += splitter.feed(c)
    return out + splitter.flush()


def test_splitter_emits_sentences_as_soon_as_complete():
    s = SentenceSplitter()
    assert s.feed("Bonjour, je suis J-ONE. Je") == ["Bonjour, je suis J-ONE."]
    assert s.feed(" suis un robot ! Et ") == ["Je suis un robot !"]
    assert s.flush() == ["Et"]


def test_splitter_handles_tokens_split_across_chunks():
    chunks = ["Il fait ", "beau", ". Vous", " allez bien", " ?", " Oui."]
    assert feed_all(SentenceSplitter(), chunks) == ["Il fait beau.", "Vous allez bien ?", "Oui."]


def test_splitter_keeps_decimals_and_abbreviations():
    out = feed_all(SentenceSplitter(), ["Il mesure 1.32 mètre, dit M. Dupont. Fin."])
    assert out == ["Il mesure 1.32 mètre, dit M. Dupont.", "Fin."]


def test_splitter_soft_cut_on_long_sentence():
    long = "Je vais vous expliquer quelque chose de passionnant, " * 4
    out = SentenceSplitter(soft_limit=80).feed(long)
    assert out and all(len(s) <= 81 for s in out)
    assert out[0].endswith(",")


def test_clean_removes_markdown_lists_emoji_and_urls():
    text = "**Bonjour** 😀 !\n- premier point\n- second\nVoir https://example.com"
    assert clean_for_speech(text) == "Bonjour ! premier point second Voir"


def test_clean_spells_symbols():
    assert clean_for_speech("Il fait 20°C et 50% d'humidité") == "Il fait 20 degrés et 50 pour cent d'humidité"


def test_think_stripper_removes_blocks_across_chunks():
    strip = ThinkStripper()
    chunks = ["<thi", "nk>je réfléchis", "</th", "ink>Bonjour", " <think>x</think>!"]
    assert "".join(strip.feed(c) for c in chunks) == "Bonjour !"


def test_think_stripper_passthrough():
    strip = ThinkStripper()
    assert strip.feed("a < b") + strip.feed(" ok") == "a < b ok"
