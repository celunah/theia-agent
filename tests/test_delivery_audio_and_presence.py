# pylint: disable=wildcard-import,unused-wildcard-import,undefined-variable,duplicate-code
from tests.test_support import *

from theia.voice import _RealtimePCMSource, _normalize_realtime_pcm


class AsyncBehaviorTests(AsyncBehaviorTestBase):
    async def test_multiple_choice_request_uses_embed_and_buttons(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        with patch.object(main._UserInputView, "wait", new=AsyncMock()):
            result = await server._request_user_input(
                channel,
                7,
                {
                    "questions": [
                        {
                            "id": "color",
                            "header": "Color",
                            "question": "Choose a color",
                            "options": [{"label": "Red"}, {"label": "Blue"}],
                        }
                    ]
                },
            )

        self.assertEqual(result, {"answers": {}})
        self.assertEqual(channel.sent[0]["embed"].title, "Choose an option")
        self.assertNotIn("content", channel.sent[0])
        self.assertEqual(
            [getattr(item, "label", None) for item in channel.sent[0]["view"].children],
            ["Red", "Blue"],
        )

    async def test_multiple_questions_are_asked_in_order_and_return_structured_answers(
        self,
    ) -> None:
        view = main._UserInputView(
            7,
            [
                {
                    "id": "color",
                    "header": "Color",
                    "question": "Choose a color",
                    "options": [{"label": "Red"}, {"label": "Blue"}],
                },
                {
                    "id": "details",
                    "header": "Details",
                    "question": "Add details",
                },
            ],
        )

        first = view.message_kwargs()
        self.assertEqual(first["embed"].title, "Choose an option")
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Red", "Blue"],
        )
        self.assertFalse(view._record_answer("Red"))
        self.assertEqual(view.question_index, 1)
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children], ["Answer"]
        )
        next_question = view.message_kwargs(for_edit=True)
        self.assertIsNone(next_question["embed"])
        self.assertTrue(next_question["content"].startswith("-# "))

        self.assertTrue(view._record_answer("A warm color"))
        self.assertEqual(
            view.value,
            {
                "answers": {
                    "color": {"answers": ["Red"]},
                    "details": {"answers": ["A warm color"]},
                }
            },
        )
        self.assertNotIn(
            "Answer all (JSON)",
            [getattr(item, "label", None) for item in view.children],
        )

    async def test_free_text_request_stays_plain_text(self) -> None:
        server = main.CodexAppServer()
        channel = _Channel()
        with patch.object(main._UserInputView, "wait", new=AsyncMock()):
            await server._request_user_input(
                channel,
                7,
                {
                    "questions": [
                        {
                            "id": "details",
                            "header": "Details",
                            "question": "Add details",
                            "isOther": True,
                        }
                    ]
                },
            )

        self.assertTrue(channel.sent[0]["content"].startswith("-# "))
        self.assertNotIn("embed", channel.sent[0])

    async def test_long_response_gets_component_pagination(self) -> None:
        calls: list[dict] = []
        message = _Message()

        async def send(**kwargs):
            calls.append(kwargs)
            return message

        await main.send_paginated(send, "x" * 4001, owner_id=7)
        self.assertEqual(len(calls), 1)
        self.assertIsNotNone(calls[0]["view"])
        self.assertNotIn("embed", calls[0])
        self.assertLessEqual(len(calls[0]["content"]), 1900)
        self.assertEqual("".join(calls[0]["view"].pages), "x" * 4001)

        response = ("line\n\n" * 1000) + "end"
        pages = main._split_pages(response)
        self.assertEqual("".join(pages), response)

    async def test_normal_response_is_plain_and_preserves_content(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        response = "This is the complete Codex response."
        await main.send_paginated(send, response, view=None)
        self.assertEqual(calls[0]["content"], response)
        self.assertNotIn("embed", calls[0])
        self.assertNotIn("view", calls[0])

    async def test_generated_images_get_only_a_follow_up_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "generated.png"
            image_path.write_bytes(b"image")
            calls: list[dict[str, Any]] = []
            sent_messages: list[Any] = []

            async def send(**kwargs: Any) -> Any:
                calls.append(kwargs)
                message = _ImageMessage() if "files" in kwargs else _Message()
                sent_messages.append(message)
                return message

            on_action = AsyncMock()
            delivery = main._ResponseDelivery(
                send,
                {},
                owner_id=7,
                channel=_Channel(),
                image_path_resolver=lambda _item: image_path,
                on_image_action=on_action,
            )
            await delivery.on_event(
                "item_completed",
                {
                    "type": "imageGeneration",
                    "id": "image-1",
                    "savedPath": str(image_path),
                },
            )
            await delivery.finalize("Here is the image.")

        image_call = next(call for call in calls if "files" in call)
        self.assertEqual(image_call["content"], "Here is the image.")
        self.assertEqual(image_call["files"][0].filename, "theia-image-1.png")
        self.assertEqual(len(image_call["files"]), 1)
        view = image_call["view"]
        self.assertIsInstance(view, main._ImageResultView)
        self.assertEqual(
            [getattr(item, "label", None) for item in view.children],
            ["Follow up"],
        )
        self.assertFalse(cast(_ImageMessage, sent_messages[-1]).edits)

    async def test_image_follow_up_edits_original_view_and_keeps_thinking_separate(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "generated.png"
            image_path.write_bytes(b"image")
            original = _ImageMessage()
            status = _Message()
            calls: list[dict[str, Any]] = []

            async def send(**kwargs: Any) -> Any:
                calls.append(kwargs)
                return status

            view = main._ImageResultView(
                7,
                (image_path,),
                on_action=AsyncMock(),
                channel=_Channel(),
            )
            view.message = original
            view.message_id = original.id
            delivery = main._ResponseDelivery(
                send,
                {},
                owner_id=7,
                image_message=original,
                image_view=view,
                existing_image_paths=(image_path,),
            )
            await delivery.on_event("item_started", {"type": "commandExecution"})
            await delivery.finalize("The follow-up is complete.")

        self.assertEqual(calls[0]["content"], "-# Thinking")
        self.assertIsNot(cast(Any, delivery.status_message), original)
        self.assertEqual(original.edits[-1]["content"], "The follow-up is complete.")
        self.assertIs(original.edits[-1]["view"], view)
        self.assertNotIn("attachments", original.edits[-1])

    async def test_image_follow_up_control_opens_the_shared_prompt_modal(self) -> None:
        image_path = Path("/tmp/generated.png")
        on_action = AsyncMock()
        view = main._ImageResultView(
            7,
            (image_path,),
            on_action=on_action,
            channel=_Channel(),
        )
        interaction = SimpleNamespace(
            user=SimpleNamespace(id=7),
            response=SimpleNamespace(send_modal=AsyncMock()),
        )

        await cast(Any, view.children[0]).callback(interaction)

        interaction.response.send_modal.assert_awaited_once()
        self.assertIsInstance(
            interaction.response.send_modal.await_args.args[0], main._PromptModal
        )
        await view._follow_up_submit(
            cast(discord.Interaction, interaction), "make it brighter"
        )
        on_action.assert_awaited_once_with(
            interaction,
            "make it brighter",
            (image_path,),
            view,
        )

    async def test_status_does_not_render_path_from_intermediate_message(self) -> None:
        message = _Message()
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return message

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.start()
        await delivery.on_event(
            "item_completed",
            {
                "type": "agentMessage",
                "phase": "commentary",
                "text": "I will inspect `src/main.py` next.",
            },
        )
        self.assertNotIn("src/main.py", calls[-1]["content"])
        self.assertTrue(calls[-1]["content"].startswith("-# "))

    async def test_intermediates_are_not_streamed_before_item_completion(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.on_event(
            "agent_message",
            {"phase": "commentary", "text": "The partial preamble"},
        )
        await delivery.on_event(
            "item_started",
            {"type": "agentMessage", "phase": "commentary", "text": "The partial"},
        )
        self.assertEqual(calls, [])

        full = "The complete preamble and intermediate message."
        await delivery.on_event(
            "item_completed",
            {"type": "agentMessage", "phase": "commentary", "text": full},
        )

        self.assertEqual(calls[0]["content"], f"-# {full}")

    async def test_thread_opening_uses_the_codex_intermediate_delivery_path(
        self,
    ) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.on_event(
            "thread_opening",
            {
                "type": "agentMessage",
                "phase": "commentary",
                "text": "I am opening a guided thread response.",
            },
        )

        self.assertEqual(
            calls[0]["content"], "-# I am opening a guided thread response."
        )

    async def test_no_tool_turn_does_not_show_thinking_or_completion(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.start()
        await delivery.finalize("The complete response.")

        self.assertEqual(
            [call["content"] for call in calls], ["The complete response."]
        )

    async def test_failed_turn_shows_reason_in_error_embed(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        await delivery.finalize(
            "Codex could not complete this request.",
            failed=True,
            error_reason="Codex turn failed: service unavailable",
        )

        self.assertEqual(calls[0]["embed"].title, "Request failed")
        self.assertIn("Reason: service unavailable", calls[0]["embed"].description)
        self.assertNotIn("content", calls[0])

    async def test_request_failure_reason_reaches_error_embed(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        with patch.object(
            main.bot.codex,
            "ask",
            AsyncMock(
                side_effect=main.CodexAppServerError("Codex turn failed: quota reached")
            ),
        ):
            await main.handle_request(
                send,
                "hello",
                channel=_Channel(),
                user_id=7,
            )

        self.assertEqual(calls[0]["embed"].title, "Request failed")
        self.assertIn("Reason: quota reached", calls[0]["embed"].description)

    async def test_request_prompt_includes_trusted_current_author_metadata(
        self,
    ) -> None:
        captured: dict[str, Any] = {}

        async def ask(prompt: str, **_kwargs: Any) -> str:
            captured["prompt"] = prompt
            return "response"

        user = SimpleNamespace(id=7, display_name="M Λ J Λ N")
        with patch.object(main.bot.codex, "ask", new=ask):
            await main.handle_request(
                AsyncMock(),
                "hello",
                channel=_Channel(),
                user_id=user.id,
                user=user,
            )

        self.assertIn("trusted Discord metadata", captured["prompt"])
        self.assertIn("Current request author user id: 7", captured["prompt"])
        self.assertIn(
            "Current request author display name: M Λ J Λ N", captured["prompt"]
        )
        self.assertTrue(captured["prompt"].endswith("hello"))

    async def test_agent_created_thread_receives_the_first_response(self) -> None:
        source = _Channel()
        source.guild = SimpleNamespace(id=1)
        thread = _Channel()
        thread.guild = source.guild

        async def ask(*_args, **kwargs):
            kwargs["on_channel_change"](thread)
            return "The first response belongs in the new thread."

        with patch.object(main.bot.codex, "ask", new=ask):
            await main.handle_request(
                source.send,
                "Create a thread for this request.",
                channel=source,
                user_id=7,
            )

        self.assertEqual(source.sent, [])
        self.assertEqual(
            thread.sent[0]["content"],
            "The first response belongs in the new thread.",
        )

    async def test_voice_mode_speaks_the_final_response_and_keeps_text(self) -> None:
        calls: list[dict] = []

        async def send(**kwargs):
            calls.append(kwargs)
            return _Message()

        speak = AsyncMock()
        with patch.object(
            main.bot.codex,
            "ask",
            AsyncMock(return_value="The complete voice response."),
        ):
            await main.handle_request(
                send,
                "hello",
                channel=_Channel(),
                user_id=7,
                speak_text=speak,
            )

        speak.assert_awaited_once_with("The complete voice response.")
        self.assertEqual(calls[-1]["content"], "The complete voice response.")
        self.assertNotIn("embed", calls[-1])

    async def test_tool_turn_shows_thinking_only_while_active(self) -> None:
        calls: list[dict] = []
        status_message = _Message()

        async def send(**kwargs):
            calls.append(kwargs)
            return status_message if len(calls) == 1 else _Message()

        delivery = main._ResponseDelivery(send, {}, owner_id=7)
        with patch("theia.delivery.time.monotonic", side_effect=[100.0, 165.0]):
            await delivery.start()
            await delivery.on_event("item_started", {"type": "commandExecution"})
            await delivery.finalize("The complete response.")

        self.assertEqual(calls[0]["content"], "-# Thinking")
        self.assertFalse(status_message.deleted)
        self.assertEqual(
            status_message.edits[-1]["content"],
            "-# Thought for 1 minute and 5 seconds",
        )
        self.assertEqual(calls[-1]["content"], "The complete response.")
        self.assertNotIn("completed", calls[-1]["content"].casefold())

    def test_thought_duration_switches_units(self) -> None:
        self.assertEqual(main._format_thought_duration(2), "Thought for 2 seconds")
        self.assertEqual(
            main._format_thought_duration(60),
            "Thought for 1 minute",
        )
        self.assertEqual(
            main._format_thought_duration(61),
            "Thought for 1 minute and 1 second",
        )

    async def test_presence_generation_is_ephemeral_and_has_no_tools(self) -> None:
        server = main.CodexAppServer()
        requests: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

        async def request(method: str, params: dict[str, Any], **kwargs: Any) -> dict:
            requests.append((method, params, kwargs))
            if method == "thread/start":
                return {"thread": {"id": "presence-thread"}}
            return {"turn": {"id": "presence-turn"}}

        server._request = AsyncMock(side_effect=request)
        server._ensure_running = AsyncMock()
        server._wait_for_turn = AsyncMock(
            return_value='{"activity_type":"watching","text":"reviewing"}'
        )
        with patch.object(
            server,
            "_system_instructions",
            return_value="base instructions with the selected personality",
        ):
            result = await server.generate_presence(
                "Use this task context only to choose a generic line.",
                session_key="presence-session",
            )

        self.assertEqual(
            result,
            {"activity_type": "watching", "text": "reviewing"},
        )
        self.assertEqual(
            [method for method, _, _ in requests], ["thread/start", "turn/start"]
        )
        thread_params = requests[0][1]
        self.assertTrue(thread_params["ephemeral"])
        self.assertEqual(thread_params["approvalPolicy"], "never")
        self.assertEqual(thread_params["sandbox"], "read-only")
        self.assertEqual(thread_params["runtimeWorkspaceRoots"], [])
        self.assertNotIn("dynamicTools", thread_params)
        self.assertEqual(requests[1][1]["effort"], "low")
        self.assertIn("outputSchema", requests[1][1])
        self.assertNotIn("presence-session", server._sessions)

    def test_direct_presence_text_uses_custom_activity_without_prefix(self) -> None:
        spec = main.RichPresenceManager._activity_from_result(
            {"activity_type": "none", "text": "A direct line"}
        )

        self.assertIsNotNone(spec)
        activity = main.RichPresenceManager._discord_activity(spec)
        self.assertIsInstance(activity, discord.CustomActivity)
        self.assertEqual(activity.type, discord.ActivityType.custom)
        self.assertEqual(activity.name, "A direct line")

    def test_presence_text_is_bounded_without_an_ellipsis(self) -> None:
        spec = main.RichPresenceManager._activity_from_result(
            {"activity_type": "playing", "text": "x" * 200}
        )

        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(len(spec.text), 128)
        self.assertFalse(spec.text.endswith("..."))

    def test_current_presence_line_is_read_only(self) -> None:
        manager = main.RichPresenceManager(AsyncMock(), AsyncMock())
        self.assertIsNone(manager.current_line)
        manager._current_activity = discord.CustomActivity("watching the moon")
        self.assertEqual(manager.current_line, "watching the moon")


class RealtimeVoiceTests(unittest.IsolatedAsyncioTestCase):
    def test_realtime_output_is_normalized_to_discord_pcm(self) -> None:
        raw = b"\x64\x00" * 480
        normalized = _normalize_realtime_pcm(
            raw,
            sample_rate=24000,
            num_channels=1,
        )
        self.assertEqual(len(normalized), main.VOICE_FRAME_BYTES)

        source = _RealtimePCMSource()
        source.feed(normalized)
        source.finish()
        self.assertEqual(len(source.read()), main.VOICE_FRAME_BYTES)
        self.assertEqual(source.read(), b"")
        source.cleanup()

    async def test_realtime_voice_manager_streams_input_and_output(self) -> None:
        client = SimpleNamespace(channel=SimpleNamespace(id=8))
        client.listen = lambda sink: setattr(client, "sink", sink)
        client.play = lambda source, after: setattr(client, "playing", (source, after))
        client.stop_playing = lambda: None
        client.disconnect = AsyncMock()
        guild = SimpleNamespace(id=42, voice_client=client)
        voice_channel = SimpleNamespace(id=8, guild=guild)
        text_channel = SimpleNamespace(send=AsyncMock(), guild=guild)
        realtime_start = AsyncMock()
        realtime_audio = AsyncMock()
        realtime_speech = AsyncMock()
        realtime_stop = AsyncMock(return_value=True)
        manager = main.VoiceModeManager(
            transcribe=AsyncMock(),
            synthesize=AsyncMock(return_value=()),
            realtime_available=lambda: True,
            realtime_start=realtime_start,
            realtime_audio=realtime_audio,
            realtime_speech=realtime_speech,
            realtime_stop=realtime_stop,
            realtime_authorized=lambda _session: True,
        )

        await manager.start(
            session_key="voice",
            user_id=7,
            voice_channel=cast(Any, voice_channel),
            text_channel=text_channel,
            allow_tools=True,
            on_transcript=AsyncMock(),
        )
        realtime_start.assert_awaited_once()
        manager._on_audio_packet(42, 8, 7, b"input")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        realtime_audio.assert_awaited_once_with("voice", b"input", 48000, 2)

        await manager._on_realtime_event(
            "voice",
            "output_audio",
            {
                "data": b"\x00" * 3840,
                "sample_rate": 48000,
                "num_channels": 2,
            },
        )
        self.assertIn("playing", vars(client))
        await manager._on_realtime_event(
            "voice",
            "transcript_done",
            {"role": "assistant", "text": "spoken response"},
        )
        await manager.speak_text("voice", "text response")
        realtime_speech.assert_awaited_once_with("voice", "text response")
        await manager.stop("voice")
        realtime_stop.assert_awaited_once_with("voice")


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


class RichPresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def _wait_for(self, predicate: Any) -> None:
        for _ in range(100):
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("timed out waiting for Rich Presence update")

    async def test_active_presence_maps_type_and_suppresses_duplicate_phases(
        self,
    ) -> None:
        changes: list[dict[str, Any]] = []
        generated: list[str] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            generated.append(prompt)
            return {"activity_type": "listening", "text": "reviewing"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Review the request.",
                channel_context="Recent exchange.",
            )
            await self._wait_for(lambda: len(changes) == 1)
            self.assertEqual(
                changes[-1]["activity"].type, discord.ActivityType.listening
            )
            self.assertEqual(changes[-1]["activity"].name, "reviewing")

            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await self._wait_for(lambda: len(generated) == 2)
            await manager.observe_event(
                "task", "item_completed", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
            await manager.observe_event(
                "task", "item_started", {"type": "commandExecution"}
            )
            await asyncio.sleep(0.02)
            self.assertEqual(len(generated), 2)
        finally:
            await manager.close()

    async def test_active_activity_overrides_idle_and_finishing_clears_it(self) -> None:
        changes: list[dict[str, Any]] = []

        async def change_presence(**kwargs: Any) -> None:
            changes.append(kwargs)

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            activity_type = "none" if "idle" in prompt else "playing"
            return {"activity_type": activity_type, "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
        )
        try:
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(changes))
            self.assertIsNotNone(changes[-1]["activity"])

            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
            await self._wait_for(
                lambda: (
                    changes[-1].get("activity") is not None
                    and changes[-1]["activity"].type == discord.ActivityType.playing
                )
            )
            await manager.finish_task("task", response="Done.")
            self.assertIsInstance(changes[-1]["activity"], discord.CustomActivity)
            self.assertEqual(changes[-1]["activity"].name, "available")
        finally:
            await manager.close()

    async def test_start_generates_idle_presence_immediately(self) -> None:
        generated = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **_kwargs: Any) -> dict[str, str]:
            self.assertIn("idle", prompt.casefold())
            generated.set()
            return {"activity_type": "none", "text": "available"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            timeout=1,
        )
        try:
            await manager.start()
            await asyncio.wait_for(generated.wait(), timeout=1)
        finally:
            await manager.close()

    async def test_idle_presence_uses_one_session_context_without_cross_guild_mixing(
        self,
    ) -> None:
        prompts: list[tuple[str, str | None]] = []

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(prompt: str, **kwargs: Any) -> dict[str, str]:
            prompts.append((prompt, kwargs.get("session_key")))
            return {"activity_type": "none", "text": "ready"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=1,
            clock=lambda: 1.0,
        )
        try:
            for request_id, session_key, guild_id, prompt in (
                ("first", "guild:1:channel:2:user:3", 1, "guild one task"),
                ("second", "guild:2:channel:4:user:5", 2, "guild two task"),
            ):
                await manager.begin_task(
                    request_id,
                    session_key=session_key,
                    guild_id=guild_id,
                    prompt=prompt,
                    channel_context=f"context for {prompt}",
                )
                await manager.finish_task(request_id, response=f"result for {prompt}")
            prompts.clear()
            await manager.refresh_idle()
            await self._wait_for(lambda: bool(prompts))
            self.assertEqual(prompts[0][1], "guild:2:channel:4:user:5")
            self.assertIn("guild two task", prompts[0][0])
            self.assertNotIn("guild one task", prompts[0][0])
        finally:
            await manager.close()

    async def test_finishing_a_task_cancels_its_pending_presence_request(self) -> None:
        started = asyncio.Event()
        cancelled = asyncio.Event()
        never = asyncio.Event()

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str]:
            started.set()
            try:
                await never.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return {"activity_type": "none", "text": "unreachable"}

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            active_debounce=0,
            timeout=10,
        )
        try:
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            await manager.finish_task("task")
            self.assertTrue(cancelled.is_set())
        finally:
            await manager.close()

    async def test_recent_idle_context_uses_the_longer_refresh_interval(self) -> None:
        now = [0.0]

        async def change_presence(**_kwargs: Any) -> None:
            return

        async def generate(_prompt: str, **_kwargs: Any) -> dict[str, str] | None:
            return None

        manager = main.RichPresenceManager(
            change_presence,
            generate,
            idle_interval=900,
            recent_idle_interval=600,
            context_max_age=1800,
            active_debounce=10,
            timeout=1,
            clock=lambda: now[0],
        )
        try:
            self.assertEqual(await manager._idle_delay(), 900)
            await manager.begin_task(
                "task",
                session_key="guild:1:channel:2:user:3",
                guild_id=1,
                prompt="Do work.",
                channel_context=None,
            )
            await manager.finish_task("task")
            self.assertEqual(await manager._idle_delay(), 600)
            now[0] = 1800
            self.assertEqual(await manager._idle_delay(), 900)
        finally:
            await manager.close()


class PresenceManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_bot_presence_waits_for_gateway_readiness(self) -> None:
        test_bot = main.TheiaBot()
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=False):
            await test_bot._change_presence_when_ready(status=discord.Status.idle)
        change_presence.assert_not_awaited()

    async def test_status_updates_retain_rich_activity(self) -> None:
        test_bot = main.TheiaBot()
        activity = discord.CustomActivity("reviewing")
        test_bot.rich_presence._current_activity = activity
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_presence_when_ready(status=discord.Status.dnd)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.dnd, "activity": activity},
        )

    async def test_rich_presence_preserves_the_current_status(self) -> None:
        test_bot = main.TheiaBot()
        test_bot.presence._current_status = discord.Status.idle
        activity = discord.CustomActivity("reviewing")
        change_presence = AsyncMock()
        test_bot.change_presence = change_presence
        with patch.object(test_bot, "is_ready", return_value=True):
            await test_bot._change_rich_presence(activity=activity)

        await_args = change_presence.await_args
        self.assertIsNotNone(await_args)
        assert await_args is not None
        self.assertEqual(
            await_args.kwargs,
            {"status": discord.Status.idle, "activity": activity},
        )

    async def _manager(self, now: list[float]) -> tuple[main.PresenceManager, list]:
        changes: list = []

        async def change_presence(**kwargs):
            changes.append(kwargs["status"])

        manager = main.PresenceManager(
            change_presence,
            idle_after=10,
            long_task_after=5,
            update_interval=60,
            clock=lambda: now[0],
        )
        await manager.start()
        return manager, changes

    async def test_recent_interaction_is_online_then_becomes_idle(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            self.assertEqual(changes[-1], discord.Status.online)
            now[0] = 10
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.idle)
        finally:
            await manager.close()

    async def test_only_long_running_tool_turn_becomes_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("tool-turn")
            await manager.observe_event(
                "tool-turn", "item_started", {"type": "commandExecution"}
            )
            now[0] = 5
            await manager.refresh()
            self.assertEqual(changes[-1], discord.Status.dnd)
            await manager.finish_request("tool-turn")
            self.assertEqual(changes[-1], discord.Status.online)
        finally:
            await manager.close()

    async def test_basic_turn_and_context_compaction_never_become_dnd(self) -> None:
        now = [0.0]
        manager, changes = await self._manager(now)
        try:
            await manager.touch()
            await manager.begin_request("basic-turn")
            now[0] = 6
            await manager.observe_event(
                "basic-turn", "compacted", {"reason": "context"}
            )
            await manager.refresh()
            now[0] = 11
            await manager.refresh()
            self.assertNotIn(discord.Status.dnd, changes)
        finally:
            await manager.close()


if __name__ == "__main__":
    unittest.main()
