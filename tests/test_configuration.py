# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *


class ConfigurationScriptTests(unittest.TestCase):
    def test_text_setup_only_requests_the_discord_token(self) -> None:
        prompts: list[str] = []
        output: list[str] = []
        values = collect_configuration(
            input_fn=lambda prompt: prompts.append(prompt) or "",
            secret_input_fn=lambda _prompt: "discord-token",
            output_fn=output.append,
        )

        self.assertEqual(values.mode, TEXT_MODE)
        self.assertEqual(
            values.as_environment(),
            {"TOKEN": "discord-token", "THEIA_DEFAULT_MODE": TEXT_MODE},
        )
        self.assertEqual(prompts, ["Mode [1/text]: "])
        self.assertNotIn("discord-token", "\n".join(output))

    def test_voice_setup_requests_both_audio_services(self) -> None:
        prompts: list[str] = []
        inputs = iter(
            [
                "2",
                "https://stt.example/v1",
                "https://tts.example/v1",
                "local-whisper",
                "local-tts",
                "voice-one",
                "wav",
            ]
        )
        secrets = iter(["discord-token", "stt-token", "tts-token"])
        values = collect_configuration(
            input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
            secret_input_fn=lambda _prompt: next(secrets),
            output_fn=lambda _message: None,
        )

        self.assertEqual(values.mode, VOICE_MODE)
        self.assertEqual(
            values.as_environment(),
            {
                "TOKEN": "discord-token",
                "THEIA_DEFAULT_MODE": VOICE_MODE,
                "STT_BASE_URL": "https://stt.example/v1",
                "STT_TOKEN": "stt-token",
                "STT_MODEL": "local-whisper",
                "TTS_BASE_URL": "https://tts.example/v1",
                "TTS_TOKEN": "tts-token",
                "TTS_MODEL": "local-tts",
                "TTS_VOICE": "voice-one",
                "TTS_FORMAT": "wav",
            },
        )
        self.assertEqual(
            prompts,
            [
                "Mode [1/text]: ",
                "STT URL (blank for Codex Realtime): ",
                "TTS URL (blank for Codex Realtime): ",
                "STT model [whisper-1]: ",
                "TTS model [tts-1]: ",
                "TTS voice [alloy]: ",
                "TTS format [mp3]: ",
            ],
        )

    def test_voice_setup_can_use_codex_realtime_without_custom_audio(self) -> None:
        prompts: list[str] = []
        inputs = iter(["2", "", "", "realtime-model", "marin"])
        values = collect_configuration(
            input_fn=lambda prompt: prompts.append(prompt) or next(inputs),
            secret_input_fn=lambda _prompt: "discord-token",
            output_fn=lambda _message: None,
        )

        self.assertEqual(values.mode, VOICE_MODE)
        self.assertEqual(values.stt_base_url, "")
        self.assertEqual(values.tts_base_url, "")
        self.assertEqual(values.realtime_model, "realtime-model")
        self.assertEqual(values.realtime_voice, "marin")
        self.assertEqual(
            values.as_environment(),
            {
                "TOKEN": "discord-token",
                "THEIA_DEFAULT_MODE": VOICE_MODE,
                "STT_BASE_URL": "",
                "STT_TOKEN": "",
                "TTS_BASE_URL": "",
                "TTS_TOKEN": "",
                "THEIA_REALTIME_MODEL": "realtime-model",
                "THEIA_REALTIME_VOICE": "marin",
            },
        )
        self.assertEqual(
            prompts,
            [
                "Mode [1/text]: ",
                "STT URL (blank for Codex Realtime): ",
                "TTS URL (blank for Codex Realtime): ",
                "Realtime model (blank for Codex default): ",
                "Realtime voice (blank for Codex default): ",
            ],
        )

    def test_setup_preserves_unrelated_dotenv_settings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# Keep this setting\nCUSTOM_SETTING=preserve-me\n"
                "TOKEN=old-token\nSTT_BASE_URL=https://old.example\n",
                encoding="utf-8",
            )
            values = validate_configuration(
                discord_token="new-token",
                mode=TEXT_MODE,
            )
            save_configuration(values, path=path)
            contents = path.read_text(encoding="utf-8")

        self.assertIn("CUSTOM_SETTING=preserve-me", contents)
        self.assertIn('TOKEN="new-token"', contents)
        self.assertIn('THEIA_DEFAULT_MODE="text"', contents)
        self.assertIn("STT_BASE_URL=https://old.example", contents)
        self.assertNotIn("old-token", contents)

    def test_setup_rejects_invalid_or_credential_bearing_audio_urls(self) -> None:
        with self.assertRaisesRegex(ConfigurationError, "HTTP or HTTPS"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="file:///tmp/stt",
                tts_base_url="https://tts.example/v1",
            )
        with self.assertRaisesRegex(ConfigurationError, "embedded credentials"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="https://user:password@stt.example/v1",
                tts_base_url="https://tts.example/v1",
            )
        with self.assertRaisesRegex(ConfigurationError, "both audio service URLs"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="https://stt.example/v1",
            )
        with self.assertRaisesRegex(ConfigurationError, "TTS format"):
            validate_configuration(
                discord_token="discord-token",
                mode=VOICE_MODE,
                stt_base_url="https://stt.example/v1",
                tts_base_url="https://tts.example/v1",
                tts_format="not-audio",
            )
