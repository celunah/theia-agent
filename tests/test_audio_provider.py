# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable
from tests.test_support import *

from theia.audio_provider import AudioProviderCapabilities, AudioProviderEvent
from theia.server.realtime import select_audio_provider


class FakeFullDuplexProvider:
    name = "qwen"
    capabilities = AudioProviderCapabilities(
        semantic_audio_understanding=True,
    )

    def __init__(self) -> None:
        self.callbacks: dict[str, Any] = {}
        self.audio: list[tuple[str, bytes, int, int]] = []
        self.speech: list[tuple[str, str]] = []
        self.interruptions: list[str] = []
        self.stopped: list[str] = []
        self.send_started = asyncio.Event()
        self.send_cancelled = asyncio.Event()
        self.block_audio = False

    @property
    def available(self) -> bool:
        return True

    async def start(self, session_key: str, on_event: Any) -> None:
        self.callbacks[session_key] = on_event

    async def send_audio(
        self, session_key: str, pcm: bytes, sample_rate: int, num_channels: int
    ) -> None:
        self.audio.append((session_key, pcm, sample_rate, num_channels))
        self.send_started.set()
        if self.block_audio:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                self.send_cancelled.set()
                raise

    async def speak_text(self, session_key: str, text: str) -> None:
        self.speech.append((session_key, text))

    async def interrupt(self, session_key: str) -> None:
        self.interruptions.append(session_key)

    async def stop(self, session_key: str) -> None:
        self.stopped.append(session_key)
        self.callbacks.pop(session_key, None)

    async def emit(self, session_key: str, event: AudioProviderEvent) -> None:
        await self.callbacks[session_key](event)


class AudioProviderTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _voice_objects() -> tuple[Any, Any, Any, Any]:
        client = SimpleNamespace(channel=SimpleNamespace(id=8))
        client.listen = lambda sink: setattr(client, "sink", sink)
        client.play = lambda source, after: setattr(client, "playing", (source, after))
        client.stop_playing = Mock()
        client.disconnect = AsyncMock()
        guild = SimpleNamespace(id=42, voice_client=client)
        voice_channel = SimpleNamespace(id=8, guild=guild)
        text_channel = SimpleNamespace(send=AsyncMock(), guild=guild)
        return client, guild, voice_channel, text_channel

    async def _manager(
        self, provider: FakeFullDuplexProvider, *, provider_name: str = "qwen"
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        client, guild, voice_channel, text_channel = self._voice_objects()
        manager = main.VoiceModeManager(
            transcribe=AsyncMock(),
            synthesize=AsyncMock(return_value=()),
            audio_provider=provider,
            provider_name=lambda: provider_name,
            realtime_authorized=lambda _session: True,
        )
        session = await manager.start(
            session_key="voice",
            user_id=7,
            voice_channel=cast(Any, voice_channel),
            text_channel=text_channel,
            allow_tools=True,
            on_transcript=AsyncMock(),
        )
        return manager, session, client, guild, voice_channel, text_channel

    async def test_fake_provider_interrupts_output_on_barge_in(self) -> None:
        provider = FakeFullDuplexProvider()
        manager, _session, client, _guild, _voice, _channel = await self._manager(
            provider
        )

        await provider.emit(
            "voice",
            AudioProviderEvent(
                "audio_output",
                audio=b"\x00" * 3840,
                sample_rate=48000,
                num_channels=2,
            ),
        )
        await provider.emit("voice", AudioProviderEvent("speech_started"))

        self.assertEqual(provider.interruptions, ["voice"])
        client.stop_playing.assert_called_once_with()
        await manager.speak_text("voice", "Theia response")
        self.assertEqual(provider.speech, [("voice", "Theia response")])
        await manager.stop("voice")

    async def test_partial_transcript_waits_for_final_submission(self) -> None:
        provider = FakeFullDuplexProvider()
        manager, session, _client, _guild, _voice, channel = await self._manager(
            provider
        )
        callback = AsyncMock()
        session.on_transcript = callback

        await provider.emit(
            "voice", AudioProviderEvent("transcript_partial", text="hel")
        )
        self.assertEqual(session.partial_transcript, "hel")
        callback.assert_not_awaited()
        await provider.emit(
            "voice", AudioProviderEvent("transcript_final", text="hello there")
        )

        callback.assert_awaited_once_with(session, "Voice input", "hello there")
        self.assertEqual(session.partial_transcript, "")
        self.assertTrue(channel.send.await_count >= 1)
        await manager.stop("voice")

    async def test_qwen_final_transcription_replaces_local_segment_stt(self) -> None:
        provider = FakeFullDuplexProvider()
        manager, _session, _client, _guild, _voice, _channel = await self._manager(
            provider
        )

        await manager._on_segment(main.VoiceSegment(42, 8, 7, "Speaker", b"wav-data"))

        cast(AsyncMock, manager._transcribe).assert_not_awaited()
        await manager.stop("voice")

    async def test_provider_disconnect_stops_the_session_and_reports_failure(
        self,
    ) -> None:
        provider = FakeFullDuplexProvider()
        manager, _session, _client, _guild, _voice, channel = await self._manager(
            provider
        )

        await provider.emit(
            "voice",
            AudioProviderEvent("provider_error", reason="provider disconnected"),
        )

        self.assertFalse(manager.has_session("voice"))
        self.assertEqual(provider.stopped, ["voice"])
        self.assertIn(
            "provider disconnected",
            channel.send.await_args.kwargs["content"],
        )

    async def test_provider_selection_falls_back_safely(self) -> None:
        self.assertEqual(
            select_audio_provider(
                "auto",
                qwen_available=True,
                realtime_available=True,
                custom_available=True,
                custom_configured=True,
            ),
            "qwen",
        )
        self.assertEqual(
            select_audio_provider(
                "qwen",
                qwen_available=False,
                realtime_available=True,
                custom_available=True,
                custom_configured=True,
            ),
            "codex-realtime",
        )
        self.assertIsNone(
            select_audio_provider(
                "auto",
                qwen_available=False,
                realtime_available=True,
                custom_available=False,
                custom_configured=True,
            )
        )

    async def test_audio_input_isolated_between_provider_sessions(self) -> None:
        provider = FakeFullDuplexProvider()
        manager, first, _client, guild, voice_channel, _channel = await self._manager(
            provider
        )
        second_channel = SimpleNamespace(send=AsyncMock(), guild=guild)
        second = await manager.start(
            session_key="other",
            user_id=8,
            voice_channel=cast(Any, voice_channel),
            text_channel=second_channel,
            allow_tools=True,
            on_transcript=AsyncMock(),
        )

        manager._on_audio_packet(42, 8, 7, b"first")
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        self.assertEqual([item[0] for item in provider.audio], [first.session_key])
        self.assertNotEqual(first.session_key, second.session_key)
        await manager.close()

    async def test_shutdown_cancels_provider_audio_worker(self) -> None:
        provider = FakeFullDuplexProvider()
        provider.block_audio = True
        manager, _session, _client, _guild, _voice, _channel = await self._manager(
            provider
        )

        manager._on_audio_packet(42, 8, 7, b"input")
        await asyncio.wait_for(provider.send_started.wait(), 1)
        await manager.close()

        self.assertTrue(provider.send_cancelled.is_set())
        self.assertEqual(provider.stopped, ["voice"])

    def test_qwen_self_model_only_claims_configured_semantic_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "THEIA_QWEN_AUDIO_URL": "ws://qwen.test/audio",
                    "THEIA_QWEN_AUDIO_SEMANTIC_AUDIO": "false",
                    "THEIA_AUDIO_PROVIDER": "qwen",
                    "THEIA_HOME": str(Path(directory) / "theia"),
                    "THEIA_STATE": str(Path(directory) / "state.json"),
                },
            ):
                server = main.CodexAppServer()
            cast(Any, server)._qwen_audio = SimpleNamespace(
                available=True,
                capabilities=main.AudioProviderCapabilities(),
            )
            cast(Any, server)._process = SimpleNamespace(returncode=None)
            cast(Any, server)._reader_task = SimpleNamespace(done=lambda: False)
            snapshot = server._self_model_snapshot(
                server._session("provider-self-model"),
                allow_tools=True,
                allow_discord_tools=False,
            )

        self.assertEqual(snapshot["active_voice_provider"], "qwen")
        self.assertEqual(snapshot["semantic_audio_understanding"], "unavailable")
