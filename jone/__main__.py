"""Lancement : python -m jone [--config config.yaml] [--text] [--no-cloud] ...

--text : saisie au clavier à la place du micro (développement sur PC : pas de VAD
ni de Whisper, mais LLM, mémoire, bascule, TTS et LED sont ceux du robot).
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .config import load_config, resolve_path

log = logging.getLogger("jone")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="jone", description="Interaction vocale du robot J-ONE")
    p.add_argument("--config", default="config.yaml", help="fichier de configuration")
    p.add_argument("--text", action="store_true", help="mode texte (clavier au lieu du micro)")
    p.add_argument("--lang", default="fr", choices=["fr", "en"],
                   help="mode texte : langue de la voix (en vocal, elle est détectée)")
    p.add_argument("--no-cloud", action="store_true", help="forcer le LLM local (test ENF-03)")
    p.add_argument("--no-local", action="store_true", help="désactiver le LLM de secours")
    p.add_argument("--output", choices=["g1", "sounddevice", "none"],
                   help="surcharger audio_output.backend")
    p.add_argument("--input", choices=["g1_multicast", "sounddevice"],
                   help="surcharger audio_input.backend")
    p.add_argument("--led", choices=["g1", "console", "none"], help="surcharger led.backend")
    return p.parse_args(argv)


def build_pipeline(cfg: dict, text_mode: bool = False):
    from .audio_output import create_speaker
    from .conversation import Conversation, load_system_prompt
    from .exchange_log import ExchangeLogger
    from .led import create_led
    from .llm.router import create_router
    from .pipeline import VoicePipeline
    from .tts import MultiVoiceTTS

    llm_cfg = cfg["llm"]
    router = create_router(cfg)
    router.warmup()

    kwargs = {}
    if not text_mode:
        from .audio_input import create_mic
        from .stt import WhisperSTT
        from .text_filter import TranscriptFilter, WakeWordGate
        from .vad import SileroVAD, SpeechSegmenter

        a = cfg["audio_input"]
        kwargs = dict(
            mic=create_mic(cfg),
            segmenter=SpeechSegmenter.from_config(SileroVAD(a["sample_rate"]), cfg),
            stt=WhisperSTT.from_config(cfg),
            text_filter=TranscriptFilter.from_config(cfg),
            wake=WakeWordGate.from_config(cfg),
            speech_end_delay_s=cfg["vad"]["min_silence_ms"] / 1000.0,
        )

    return VoicePipeline(
        router=router,
        conversation=Conversation(llm_cfg["history_max_turns"]),
        system_prompt=load_system_prompt(resolve_path(llm_cfg["system_prompt_file"])),
        tts=MultiVoiceTTS.from_config(cfg),
        speaker=create_speaker(cfg),
        led=create_led(cfg),
        exchange_log=ExchangeLogger(resolve_path(cfg["logging"]["dir"])),
        max_sentences=llm_cfg["max_sentences"],
        tail_margin_s=cfg["audio_output"]["tail_margin_s"],
        error_hold_s=cfg["led"]["error_hold_s"],
        **kwargs,
    )


def run_text_mode(pipeline, language: str = "fr") -> None:
    from .led import State

    print("Mode texte : tapez votre phrase (« /reset » pour effacer la mémoire, Ctrl+C pour quitter).")
    pipeline.led.set_state(State.LISTENING)
    while True:
        try:
            line = input("\nVous > ").strip()
        except EOFError:
            return
        if not line:
            continue
        if line == "/reset":
            pipeline.conversation.clear()
            print("(mémoire effacée)")
            continue
        try:
            reply = pipeline.respond(line, language=language)
            print(f"J-ONE > {reply}")
        except Exception:
            log.exception("Erreur pendant la réponse")
        pipeline.led.set_state(State.LISTENING)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.no_cloud:
        cfg["llm"]["cloud"]["enabled"] = False
    if args.no_local:
        cfg["llm"]["local"]["enabled"] = False
    if args.output:
        cfg["audio_output"]["backend"] = args.output
    if args.input:
        cfg["audio_input"]["backend"] = args.input
    if args.led:
        cfg["led"]["backend"] = args.led

    from .exchange_log import setup_logging

    setup_logging(resolve_path(cfg["logging"]["dir"]), cfg["logging"]["level"])
    pipeline = build_pipeline(cfg, text_mode=args.text)

    def shutdown(*_):
        pipeline.stop()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shutdown)
    try:
        if args.text:
            run_text_mode(pipeline, args.lang)
        else:
            pipeline.run()
    except KeyboardInterrupt:
        log.info("Arrêt demandé")
    finally:
        from .led import State

        pipeline.speaker.close()
        pipeline.led.set_state(State.IDLE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
