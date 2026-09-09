# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class _AudioHTTPResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return self.body


class AudioProtocolTests(unittest.IsolatedAsyncioTestCase):
    def _environment(self) -> dict[str, str]:
        return {
            "STT_PROTOCOL": "openai-compatible",
            "STT_BASE_URL": "http://transcribe.test/v1/",
            "STT_TOKEN": "transcription-secret",
            "STT_MODEL": "local-whisper",
            "THEIA_TRANSCRIPTION_PROTOCOL": "openai-compatible",
            "THEIA_TRANSCRIPTION_BASE_URL": "http://transcribe.test/v1/",
            "THEIA_TRANSCRIPTION_API_KEY": "transcription-secret",
            "THEIA_TRANSCRIPTION_MODEL": "local-whisper",
            "TTS_PROTOCOL": "openai-compatible",
            "TTS_BASE_URL": "http://speech.test/v1",
            "TTS_TOKEN": "tts-secret",
            "TTS_MODEL": "local-tts",
            "TTS_VOICE": "voice-one",
            "TTS_FORMAT": "wav",
            "THEIA_TTS_PROTOCOL": "openai-compatible",
            "THEIA_TTS_BASE_URL": "http://speech.test/v1",
            "THEIA_TTS_API_KEY": "tts-secret",
            "THEIA_TTS_MODEL": "local-tts",
            "THEIA_TTS_VOICE": "voice-one",
            "THEIA_TTS_FORMAT": "wav",
        }

    async def test_transcription_uses_its_own_openai_compatible_base_url(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            service = main.OpenAICompatibleAudio.from_environment()
            with patch(
                "theia.audio.urllib.request.urlopen",
                return_value=_AudioHTTPResponse(b'{"text":"hello from audio"}'),
            ) as urlopen:
                result = await service.transcribe(
                    "clip.ogg", b"audio-data", "audio/ogg"
                )

        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://transcribe.test/v1/audio/transcriptions",
        )
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            request.headers["Authorization"], "Bearer transcription-secret"
        )
        self.assertIn(b'name="model"', request.data)
        self.assertIn(b"local-whisper", request.data)
        self.assertIn(b'filename="clip.ogg"', request.data)
        self.assertEqual(result, "hello from audio")

    async def test_tts_uses_its_own_base_url_and_json_protocol(self) -> None:
        with patch.dict(os.environ, self._environment(), clear=False):
            service = main.OpenAICompatibleAudio.from_environment()
            with patch(
                "theia.audio.urllib.request.urlopen",
                return_value=_AudioHTTPResponse(b"wav-data"),
            ) as urlopen:
                result = await service.synthesize("hello")

        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://speech.test/v1/audio/speech")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.headers["Authorization"], "Bearer tts-secret")
        self.assertEqual(
            json.loads(request.data),
            {
                "model": "local-tts",
                "input": "hello",
                "voice": "voice-one",
                "response_format": "wav",
            },
        )
        assert result is not None
        self.assertEqual(result.data, b"wav-data")
        self.assertEqual(result.filename, "theia-response.wav")

    async def test_audio_response_can_be_attached_to_normal_text(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        await main.send_paginated(
            send,
            "normal text response",
            speech=(main.AudioOutput(b"audio", "response.mp3", "audio/mpeg"),),
        )

        self.assertEqual(calls[0]["content"], "normal text response")
        self.assertEqual(calls[0]["files"][0].filename, "response.mp3")

    async def test_configured_transcription_is_added_alongside_local_audio(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(
                os.environ,
                self._environment()
                | {
                    "THEIA_HOME": str(root / "theia"),
                    "THEIA_STATE": str(root / "state.json"),
                },
                clear=False,
            ):
                server = main.CodexAppServer()
                server._audio.transcribe = AsyncMock(return_value="spoken request")
                attachment = SimpleNamespace(
                    filename="voice.ogg",
                    content_type="audio/ogg",
                    size=5,
                    read=AsyncMock(return_value=b"audio"),
                )
                prepared = await server._prepare_attachments((attachment,))

        self.assertEqual(prepared[0]["type"], "localAudio")
        self.assertEqual(prepared[1]["type"], "text")
        self.assertIn("spoken request", prepared[1]["text"])
