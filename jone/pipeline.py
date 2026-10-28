"""Boucle principale : écoute → transcription → LLM → synthèse → lecture → réécoute.

Chaque étape est un module indépendant injecté dans `VoicePipeline` (ENF-08), ce
qui permet aussi de tester la boucle avec des doublures.
Une erreur sur un tour allume la LED en rouge, est journalisée, et la boucle
reprend l'écoute : le pipeline ne s'arrête jamais sur une exception (ENF-06).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .conversation import Conversation, load_system_prompt
from .led import Led, State
from .llm.router import LLMRouter, NoBackendAvailable, RouteInfo
from .stt import Transcription
from .text_utils import SentenceSplitter, clean_for_speech

log = logging.getLogger(__name__)

NO_LLM_REPLIES = {
    "fr": "Désolé, je n'arrive pas à réfléchir pour le moment. Réessayez dans un instant.",
    "en": "Sorry, I can't think right now. Please try again in a moment.",
}
EMPTY_REPLIES = {
    "fr": "Pardon, je n'ai pas compris. Pouvez-vous reformuler ?",
    "en": "Sorry, I didn't catch that. Could you say it again?",
}
NO_LLM_REPLY = NO_LLM_REPLIES["fr"]
EMPTY_REPLY = EMPTY_REPLIES["fr"]


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(seconds * 1000, 1)


class VoicePipeline:
    def __init__(self, *, router: LLMRouter, conversation: Conversation, system_prompt: str,
                 tts, speaker, led: Led, exchange_log, max_sentences: int = 3,
                 tail_margin_s: float = 0.4, error_hold_s: float = 2.0,
                 mic=None, segmenter=None, stt=None, text_filter=None, wake=None,
                 speech_end_delay_s: float = 0.0, clock: Callable[[], float] = time.perf_counter):
        self.router = router
        self.conversation = conversation
        self.system_prompt = system_prompt
        self.tts = tts
        self.speaker = speaker
        self.led = led
        self.exchange_log = exchange_log
        self.max_sentences = max_sentences
        self.tail_margin_s = tail_margin_s
        self.error_hold_s = error_hold_s
        self.mic = mic
        self.segmenter = segmenter
        self.stt = stt
        self.text_filter = text_filter
        self.wake = wake
        # Le VAD ne déclare la fin de parole qu'après `min_silence_ms` de silence :
        # on l'ajoute pour mesurer la latence depuis la vraie fin de parole (ENF-01).
        self.speech_end_delay_s = speech_end_delay_s
        self.clock = clock
        self._stop = threading.Event()

    # ------------------------------------------------------------------ boucle
    def run(self) -> None:
        self.mic.start()
        self._listen()
        log.info("J-ONE à l'écoute")
        try:
            while not self._stop.is_set():
                frame = self.mic.read(timeout=0.5)
                if frame is None:
                    continue
                utterance = self.segmenter.process(frame)
                if utterance is None:
                    continue
                detected_at = self.clock()
                try:
                    self.handle_utterance(utterance, detected_at)
                except Exception:
                    log.exception("Erreur pendant le traitement d'un énoncé")
                    self.speaker.stop()
                    self._show_error()
                finally:
                    self._listen()
        finally:
            self.mic.close()

    def stop(self) -> None:
        self._stop.set()

    def _listen(self) -> None:
        """Reprise de l'écoute (EF-10) : tampons vidés pour ne rien garder de la
        voix du robot, puis LED verte."""
        self.segmenter.reset()
        self.mic.resume()
        self.led.set_state(State.LISTENING)

    def _show_error(self) -> None:
        self.led.set_state(State.ERROR)
        time.sleep(self.error_hold_s)

    # ------------------------------------------------------------ un énoncé
    def handle_utterance(self, audio, detected_at: float) -> None:
        self.mic.pause()  # plus d'écoute jusqu'à la fin de la réponse (EF-10)
        self.led.set_state(State.THINKING)
        speech_end = detected_at - self.speech_end_delay_s

        tr: Transcription = self.stt.transcribe(audio)
        record = {
            "transcription": tr.text,
            "audio_s": round(tr.audio_s, 2),
            "langue": tr.language,
            "stt": {"avg_logprob": round(tr.avg_logprob, 3),
                    "no_speech_prob": round(tr.no_speech_prob, 3),
                    "compression_ratio": round(tr.compression_ratio, 2)},
            "latences_ms": {"stt": round(tr.elapsed_ms, 1)},
        }

        verdict = self.text_filter.check(tr)
        if not verdict.accepted:
            log.info("Transcription rejetée (%s) : %r", verdict.reason, tr.text)
            self.exchange_log.write({**record, "statut": "rejete", "motif": verdict.reason})
            return

        user_text = self.wake.admit(verdict.text, time.monotonic()) if self.wake else verdict.text
        if user_text is None:
            log.info("Pas de mot d'éveil, énoncé ignoré : %r", verdict.text)
            self.exchange_log.write({**record, "statut": "ignore", "motif": "pas de mot d'eveil"})
            return

        log.info("Utilisateur (%s) : %s", tr.language, user_text)
        self.respond(user_text, record, speech_end, language=tr.language)
        if self.wake:
            self.wake.mark_exchange(time.monotonic())

    # ----------------------------------------------------------- la réponse
    def respond(self, user_text: str, record: dict | None = None,
                speech_end: float | None = None, language: str = "fr") -> str:
        """Fait répondre le robot. `language` choisit la voix et les messages de
        secours ; le LLM, lui, répond dans la langue de l'interlocuteur."""
        record = dict(record or {})
        record.setdefault("transcription", user_text)
        record.setdefault("langue", language)
        latencies = record.setdefault("latences_ms", {})
        t_start = self.clock()
        speech_end = t_start if speech_end is None else speech_end

        self.led.set_state(State.THINKING)
        self.speaker.begin_response()
        info = RouteInfo()
        spoken: list[str] = []
        tts_first: float | None = None
        status = "ok"

        def say(sentence: str) -> None:
            nonlocal tts_first
            text = clean_for_speech(sentence, language)
            if not text:
                return
            t0 = self.clock()
            pcm = self.tts.synthesize(text, language=language)
            if tts_first is None:
                tts_first = self.clock() - t0
                self.led.set_state(State.SPEAKING)
            self.speaker.enqueue(pcm)
            spoken.append(text)

        splitter = SentenceSplitter()
        messages = self.conversation.with_user(user_text)
        try:
            stream = self.router.stream(self.system_prompt, messages, info, language=language)
            try:
                for delta in stream:
                    for sentence in splitter.feed(delta):
                        say(sentence)
                        if len(spoken) >= self.max_sentences:  # EF-06 : 3 phrases max
                            break
                    if len(spoken) >= self.max_sentences:
                        break
                else:
                    for sentence in splitter.flush():
                        if len(spoken) < self.max_sentences:
                            say(sentence)
            finally:
                stream.close()
        except NoBackendAvailable as exc:
            log.error("Aucun LLM disponible : %s", exc)
            status = "erreur_llm"
            say(NO_LLM_REPLIES.get(language, NO_LLM_REPLY))

        if not spoken:
            status = "reponse_vide" if status == "ok" else status
            say(EMPTY_REPLIES.get(language, EMPTY_REPLY))

        self.speaker.wait_until_done()
        first_audio = self.speaker.first_audio_at
        time.sleep(self.tail_margin_s)  # écho résiduel de la pièce avant de réécouter

        if status == "erreur_llm":
            self._show_error()

        reply = " ".join(spoken)
        if status == "ok":
            self.conversation.commit(user_text, reply)
        log.info("J-ONE (%s) : %s", info.backend or "-", reply)

        latencies.update({
            "llm_premier_jeton": _ms(info.first_token_s),
            "tts_premiere_phrase": _ms(tts_first),
            "bout_en_bout": _ms(first_audio - speech_end) if first_audio else None,
            "total": _ms(self.clock() - speech_end),
        })
        self.exchange_log.write({
            **record,
            "statut": status,
            "reponse": reply,
            "llm": info.backend,
            "bascule": info.fallback,
            "erreurs_llm": info.errors,
            "latences_ms": latencies,
        })
        return reply
