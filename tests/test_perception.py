import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from theia.server.core import CodexAppServer
from theia.server.perception import (
    CodexModalityCapabilities,
    PerceptionError,
    PerceptionReport,
    QwenPerceptionClient,
    Speaker,
    TimelineEvent,
    TranscriptSegment,
    route_attachments,
)


def _attachment(filename: str, content_type: str, *, size: int = 10) -> SimpleNamespace:
    return SimpleNamespace(filename=filename, content_type=content_type, size=size)


def _report() -> PerceptionReport:
    return PerceptionReport(
        schema_version="1.0",
        media_type="video",
        duration_seconds=4.0,
        languages=("en",),
        topics=("test",),
        speakers=(Speaker(speaker_id="speaker-1"),),
        transcript_segments=(TranscriptSegment("speaker-1", 0.0, 1.0, "hello"),),
        timeline_events=(TimelineEvent(0.0, 1.0, "visual_event", "A test frame"),),
        visual_observations=("A test frame",),
        audio_observations=(),
        summary="A test media report.",
        confidence=0.9,
        uncertainties=(),
    )


class PerceptionTests(unittest.IsolatedAsyncioTestCase):
    def test_capability_snapshot_is_not_extension_only(self) -> None:
        capabilities = CodexModalityCapabilities.from_snapshot(
            {"inputModalities": ["text", "image"], "maxInputBytes": 100}
        )
        image = _attachment("not-an-image-name", "image/png", size=10)
        video = _attachment("clip.bin", "video/mp4", size=10)

        route = route_attachments((image, video), capabilities, qwen_available=True)

        self.assertEqual(route.native_indices, (0,))
        self.assertEqual(route.qwen_indices, (1,))

    def test_codex_size_limit_routes_media_to_qwen(self) -> None:
        capabilities = CodexModalityCapabilities.from_snapshot(
            {"inputModalities": ["image"], "maxFileSize": 10}
        )
        route = route_attachments(
            (_attachment("image.png", "image/png", size=11),),
            capabilities,
            qwen_available=True,
        )

        self.assertEqual(route.native_indices, ())
        self.assertEqual(route.qwen_indices, (0,))

    def test_dedicated_request_routes_native_media_only_once(self) -> None:
        capabilities = CodexModalityCapabilities.from_snapshot(
            {"inputModalities": ["image"]}
        )
        route = route_attachments(
            (_attachment("image.png", "image/png"),),
            capabilities,
            qwen_available=True,
            dedicated_requested=True,
        )

        self.assertEqual(route.native_indices, ())
        self.assertEqual(route.qwen_indices, (0,))
        self.assertEqual(set(route.native_indices) & set(route.qwen_indices), set())

    async def test_native_capability_skips_qwen_and_keeps_media_for_codex(self) -> None:
        server = CodexAppServer()
        client = SimpleNamespace(
            available=True,
            perceive_attachments=AsyncMock(return_value=_report()),
        )
        cast(Any, server)._qwen_perception_client = client
        server.provider_capabilities = AsyncMock(
            return_value={"inputModalities": ["image", "audio"]}
        )
        image = _attachment("photo.dat", "image/png")

        native, context = await server._prepare_perception_context(
            "Describe this.", (image,), dedicated_requested=False
        )

        self.assertEqual(native, (image,))
        self.assertEqual(context, "")
        client.perceive_attachments.assert_not_awaited()

    async def test_unsupported_media_is_sent_only_to_qwen_and_injected(self) -> None:
        server = CodexAppServer()
        client = SimpleNamespace(
            available=True,
            perceive_attachments=AsyncMock(return_value=_report()),
        )
        cast(Any, server)._qwen_perception_client = client
        server.provider_capabilities = AsyncMock(
            return_value={"inputModalities": ["image"]}
        )
        image = _attachment("photo.png", "image/png")
        video = _attachment("clip.bin", "video/mp4")

        native, context = await server._prepare_perception_context(
            "What happens?", (image, video), dedicated_requested=False
        )

        self.assertEqual(native, (image,))
        self.assertIn("[perception report]", context)
        client.perceive_attachments.assert_awaited_once_with(
            (video,), instruction="What happens?"
        )

    def test_malformed_report_never_becomes_codex_context(self) -> None:
        with self.assertRaises(PerceptionError):
            PerceptionReport.from_payload(
                {
                    "schema_version": "1.0",
                    "media_type": "video",
                    "confidence": 3,
                }
            )

    async def test_qwen_client_accepts_local_paths_and_records_usage(self) -> None:
        usage: list[dict[str, Any]] = []
        client = QwenPerceptionClient(
            "https://qwen.test/v1",
            "private-key",
            input_usd_per_million=1.0,
            output_usd_per_million=2.0,
            usage_callback=usage.append,
        )
        response = {
            "choices": [{"message": {"content": json.dumps(_report().to_dict())}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
        cast(Any, client)._post = Mock(return_value=response)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "clip.mp4"
            path.write_bytes(b"video")
            report = await client.perceive_attachments(
                (path,), instruction="What happens?"
            )

        self.assertEqual(report.summary, "A test media report.")
        self.assertEqual(usage[0]["modality"], "video")
        self.assertEqual(usage[0]["input_tokens"], 10)
        self.assertEqual(usage[0]["output_tokens"], 5)
        self.assertEqual(usage[0]["estimated_usd"], 0.00002)

    async def test_qwen_client_retries_malformed_response_and_reports_failure(
        self,
    ) -> None:
        client = QwenPerceptionClient("https://qwen.test/v1", "private-key")
        cast(Any, client)._post = Mock(
            side_effect=[
                {"choices": [{"message": {"content": "not json"}}]},
                {"choices": [{"message": {"content": "still not json"}}]},
            ]
        )
        attachment = SimpleNamespace(
            filename="clip.mp4",
            content_type="video/mp4",
            size=4,
            read=AsyncMock(return_value=b"data"),
        )

        with self.assertRaises(PerceptionError) as raised:
            await client.perceive_attachments((attachment,), instruction="Describe it")

        self.assertEqual(raised.exception.retry_count, 1)

    async def test_qwen_client_rejects_missing_media_and_unknown_mime_stays_native(
        self,
    ) -> None:
        client = QwenPerceptionClient("https://qwen.test/v1", "private-key")
        with self.assertRaises(PerceptionError):
            await client.perceive_attachments((), instruction="Describe it")

        route = route_attachments(
            (_attachment("document.bin", "application/pdf"),),
            CodexModalityCapabilities.from_snapshot(
                {"inputModalities": ["image", "audio", "video"]}
            ),
            qwen_available=True,
        )
        self.assertEqual(route.native_indices, (0,))
        self.assertEqual(route.qwen_indices, ())

    def test_perception_usage_is_separate_from_codex_tokens(self) -> None:
        server = CodexAppServer()
        cast(Any, server)._persist_state = Mock()
        server._record_perception_usage(
            {
                "provider": "Alibaba Model Studio",
                "model": "qwen3.8-omni-flash",
                "modality": "video",
                "duration_sec": 2.5,
                "input_tokens": 10,
                "output_tokens": 5,
                "estimated_usd": 0.00002,
                "success": True,
                "retry_count": 1,
                "file_sizes_bytes": [1024],
                "file_durations_seconds": [4.0],
            }
        )

        perception = server.theia_usage()["perception"]
        self.assertEqual(perception["requests"], 1)
        self.assertEqual(perception["inputTokens"], 10)
        self.assertEqual(perception["outputTokens"], 5)
        self.assertEqual(perception["estimatedUsd"], 0.00002)
        self.assertEqual(perception["records"][0]["fileSizesBytes"], [1024])


if __name__ == "__main__":
    unittest.main()
