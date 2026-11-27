import pytest

from jone.dialogue import language_command


@pytest.mark.parametrize("text, expected", [
    ("Parle en anglais.", "en"),
    ("Parle en français.", "fr"),
    ("Maintenant je voulais qu'on parle en anglais si possible.", "en"),
    ("Speak French please.", "fr"),
    ("Can you speak English?", "en"),
    ("In English please.", "en"),
    ("En français s'il te plaît", "fr"),
    # pas des commandes
    ("Peux-tu switcher du français à l'anglais et vice versa ?", None),
    ("Est-ce que tu peux parler d'autres langues que le français ?", None),
    ("Oui, pas en français.", None),
    ("Comment dit-on bonjour en anglais quand on est à Casablanca ?", None),
    ("Quelle est la capitale de l'Australie ?", None),
    ("I'm going to go to the bathroom.", None),
])
def test_language_command(text, expected):
    assert language_command(text) == expected
