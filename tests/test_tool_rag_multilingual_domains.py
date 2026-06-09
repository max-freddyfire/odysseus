"""The tool-RAG domain classifier (`_classify_agent_request` in
src/agent_loop.py) only matched English keywords, so a non-English agent
request matched no domain, was flagged `low_signal=True`, and tool retrieval
was skipped entirely: the model received only the always-available tools
(manage_memory, ask_user, update_plan) and told the user it could not do the
task. Live repro (Norwegian, agent mode):

    [agent-intent] latest='Vis meg de fem siste e-postene i innboksen min.'
        continuation=False low_signal=True domains=[] ...
    [tool-rag] Low-signal agent message; skipping retrieval and using
        always-available tools only
    [agent-intent] selected_tools=['ask_user', 'manage_memory', 'update_plan']

while the English "Show me the five latest emails in my inbox." classified
domains=['email', 'ui', 'web'] and was offered every email tool.

The classifier is deterministic string matching (no embeddings / no DB), so it
can be exercised directly. The non-English cases fail on current dev and pass
with the multilingual keyword extension; the English cases pin existing
behavior, and smalltalk must stay low-signal in every language.
"""
import pytest

from src.agent_loop import _classify_agent_request


def _classify(text):
    return _classify_agent_request([{"role": "user", "content": text}], text)


# ---------------------------------------------------------------------------
# Non-English requests must classify into the same domain as their English
# equivalent and must NOT be low-signal (which would skip tool retrieval).
# Languages: Swedish / Norwegian / Danish / German / Spanish / French /
# Italian (the live repro in #3766 was Italian).
# ---------------------------------------------------------------------------
NON_ENGLISH_DOMAIN_CASES = [
    # email — the live-repro phrasing first
    ("Vis meg de fem siste e-postene i innboksen min.", "email"),
    ("Visa de senaste mejlen i min inkorg.", "email"),
    ("Vis de seneste mails i min indbakke.", "email"),
    ("Zeig mir den Posteingang.", "email"),
    ("Muéstrame los últimos correos.", "email"),
    ("Montre-moi les derniers courriels dans ma boîte de réception.", "email"),
    # notes / calendar / tasks
    ("Lag et notat om dette.", "notes_calendar_tasks"),
    ("Skapa en anteckning om mötet.", "notes_calendar_tasks"),
    ("Erstell eine Notiz und eine Erinnerung für morgen.", "notes_calendar_tasks"),
    ("Crea un recordatorio para mañana.", "notes_calendar_tasks"),
    ("Ajoute un rappel pour demain.", "notes_calendar_tasks"),
    ("Legg det inn i kalenderen min.", "notes_calendar_tasks"),
    # documents
    ("Lag et dokument med et utkast til talen.", "documents"),
    ("Skriv en dikt om havet.", "documents"),
    ("Schreib einen Entwurf für den Brief.", "documents"),
    ("Escribe un borrador del informe.", "documents"),
    ("Écris un brouillon de la lettre.", "documents"),
    # web
    ("Søk på nettet etter dagens nyheter.", "web"),
    ("Sök på nätet efter senaste nyheterna.", "web"),
    ("Such im Internet nach den Nachrichten.", "web"),
    ("Busca en internet las noticias de hoy.", "web"),
    ("Cherche sur internet la météo.", "web"),
    ("Hvordan blir været i morgen?", "web"),
    # files
    ("Vis meg filene i mappen.", "files"),
    ("Visa filerna i mappen.", "files"),
    ("Zeig mir die Dateien im Ordner.", "files"),
    ("Muéstrame los archivos de la carpeta.", "files"),
    ("Montre-moi les fichiers dans le dossier.", "files"),
    # settings
    ("Endre innstillingene for varsler.", "settings"),
    ("Ändra inställningarna för aviseringar.", "settings"),
    ("Ändere die Einstellungen für Benachrichtigungen.", "settings"),
    ("Cambia la configuración de las notificaciones.", "settings"),
    ("Modifie les paramètres des notifications.", "settings"),
    # ui
    ("Åpne gallerivisningen.", "ui"),
    ("Öppna panelen.", "ui"),
    ("Öffne das Panel.", "ui"),
    ("Abre el panel.", "ui"),
    ("Ouvre le panneau.", "ui"),
    # cookbook
    ("Last ned modellen til maskinen min.", "cookbook"),
    ("Ladda ner modellen.", "cookbook"),
    ("Lade das Modell herunter.", "cookbook"),
    ("Descarga el modelo.", "cookbook"),
    ("Télécharge le modèle.", "cookbook"),
    # Italian (live repro language in #3766)
    ("Mostrami la posta in arrivo.", "email"),
    ("Crea un promemoria per domani.", "notes_calendar_tasks"),
    ("Aggiungi l'appuntamento al calendario.", "notes_calendar_tasks"),
    ("Scrivi una bozza della lettera.", "documents"),
    ("Cerca su internet le notizie di oggi.", "web"),
    ("Mostrami i file nella cartella.", "files"),
    ("Cambia le impostazioni delle notifiche.", "settings"),
    ("Apri il pannello.", "ui"),
    ("Scarica il modello.", "cookbook"),
]


@pytest.mark.parametrize("text,domain", NON_ENGLISH_DOMAIN_CASES)
def test_non_english_request_gets_domain_and_is_not_low_signal(text, domain):
    intent = _classify(text)
    assert domain in intent["domains"], (
        f"expected {domain!r} for {text!r}, got {sorted(intent['domains'])}"
    )
    assert intent["low_signal"] is False, f"must not be low_signal: {text!r}"


# ---------------------------------------------------------------------------
# English behavior is pinned: same phrasings as before classify identically.
# ---------------------------------------------------------------------------
ENGLISH_PINNED_CASES = [
    ("Show me the five latest emails in my inbox.", "email"),
    ("Create a note about this.", "notes_calendar_tasks"),
    ("Write a poem about the sea.", "documents"),
    ("Search the web for today's news.", "web"),
    ("Show me the files in the folder.", "files"),
    ("Configure the email endpoint.", "settings"),
    ("Download the model.", "cookbook"),
]


@pytest.mark.parametrize("text,domain", ENGLISH_PINNED_CASES)
def test_english_request_classification_pinned(text, domain):
    intent = _classify(text)
    assert domain in intent["domains"], (
        f"expected {domain!r} for {text!r}, got {sorted(intent['domains'])}"
    )
    assert intent["low_signal"] is False


# ---------------------------------------------------------------------------
# Smalltalk stays low-signal in every language: no domain keywords, no
# retrieval. The fix widens keyword coverage, it must not make every
# non-English message a domain hit.
# ---------------------------------------------------------------------------
SMALLTALK_CASES = [
    "Hei, hvordan går det med deg i dag?",
    "Hej, hur mår du idag?",
    "Hallo, wie geht es dir heute?",
    "Hola, ¿cómo estás hoy?",
    "Salut, comment ça va aujourd'hui ?",
    "Hello, how are you today?",
]


# ---------------------------------------------------------------------------
# Cross-language collision guard: Italian "notizie" (news) must not trip the
# German "Notiz" (note) keyword — found in live testing.
# ---------------------------------------------------------------------------
def test_italian_news_request_is_web_only_not_notes():
    intent = _classify("Cerca su internet le notizie di oggi.")
    assert intent["domains"] == {"web"}, sorted(intent["domains"])


@pytest.mark.parametrize("text", SMALLTALK_CASES)
def test_smalltalk_stays_low_signal(text):
    intent = _classify(text)
    assert intent["domains"] == set(), (
        f"smalltalk must match no domain: {text!r} -> {sorted(intent['domains'])}"
    )
    assert intent["low_signal"] is True
