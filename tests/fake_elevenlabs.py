"""A stand-in for the ElevenLabs API, used by the automated tests. Never touches the network."""

from __future__ import annotations

from types import SimpleNamespace


class FakeElevenLabs:
    def __init__(self, transcript: str = "Who owes me money?", fail: bool = False):
        self.transcript = transcript
        self.fail = fail
        self.spoken: list[str] = []
        self.heard: list[bytes] = []
        self.text_to_speech = SimpleNamespace(convert=self._tts)
        self.speech_to_text = SimpleNamespace(convert=self._stt)

    def _tts(self, voice_id, text, model_id, output_format, **_):
        if self.fail:
            raise ConnectionError("simulated ElevenLabs outage")
        self.spoken.append(text)
        return iter([b"ID3", b"fake-mp3:", text.encode()[:40]])

    def _stt(self, file, model_id, **_):
        if self.fail:
            raise ConnectionError("simulated ElevenLabs outage")
        self.heard.append(file[1])
        return SimpleNamespace(text=self.transcript)
