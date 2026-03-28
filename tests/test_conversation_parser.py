"""Tests for multi-format conversation parser."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from dormancy_detect.parsers.conversation_parser import ConversationParser


@pytest.fixture
def parser():
    return ConversationParser()


@pytest.fixture
def tmp_file(tmp_path):
    """Helper to write content to a temp file and return its path."""
    def _write(content: str, name: str = "input.json") -> Path:
        fp = tmp_path / name
        fp.write_text(content, encoding="utf-8")
        return fp
    return _write


# -- Native JSON format ---------------------------------------------------

class TestNativeJson:
    def test_sessions_with_turns(self, parser, tmp_file):
        data = {
            "sessions": [
                {
                    "session_id": "s1",
                    "turns": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there"},
                    ],
                },
                {
                    "session_id": "s2",
                    "turns": [
                        {"role": "user", "content": "How are you?"},
                        {"role": "assistant", "content": "I'm well"},
                    ],
                },
            ]
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert len(sessions) == 2
        assert sessions[0].session_id == "s1"
        assert sessions[1].session_id == "s2"
        assert len(sessions[0].turns) == 2

    def test_list_of_sessions(self, parser, tmp_file):
        data = [
            {
                "session_id": "s1",
                "turns": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ],
            }
        ]
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert len(sessions) == 1
        assert sessions[0].session_id == "s1"

    def test_single_session_dict(self, parser, tmp_file):
        data = {
            "session_id": "s1",
            "turns": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert len(sessions) == 1

    def test_integer_session_ids_coerced_to_string(self, parser, tmp_file):
        data = {
            "sessions": [
                {
                    "session_id": 1,
                    "turns": [
                        {"role": "user", "content": "Hello"},
                    ],
                }
            ]
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert sessions[0].session_id == "1"


# -- ChatGPT format -------------------------------------------------------

class TestChatGPT:
    def test_basic_chatgpt_export(self, parser, tmp_file):
        data = {
            "title": "Test conversation",
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["What is Python?"]},
                        "create_time": 1000,
                    },
                    "parent": None,
                },
                "node2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": ["Python is a programming language."]},
                        "create_time": 1001,
                    },
                    "parent": "node1",
                },
                "node3": {
                    "message": {
                        "author": {"role": "system"},
                        "content": {"parts": ["You are a helpful assistant."]},
                        "create_time": 999,
                    },
                    "parent": None,
                },
            },
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        # System messages are excluded
        turns = sessions[0].turns
        assert sessions[0].session_id == "input"
        assert len(turns) == 2
        assert turns[0].role.value == "user"
        assert turns[1].role.value == "assistant"
        assert "Python" in turns[0].content

    def test_chatgpt_skips_empty_content(self, parser, tmp_file):
        data = {
            "mapping": {
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hello"]},
                        "create_time": 1000,
                    },
                },
                "node2": {
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"parts": []},
                        "create_time": 1001,
                    },
                },
            },
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        turns = sessions[0].turns
        assert len(turns) == 1

    def test_chatgpt_skips_null_messages(self, parser, tmp_file):
        data = {
            "mapping": {
                "root": {"message": None},
                "node1": {
                    "message": {
                        "author": {"role": "user"},
                        "content": {"parts": ["Hi"]},
                        "create_time": 1000,
                    },
                },
            },
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert len(sessions[0].turns) == 1

    def test_chatgpt_auto_detected(self, parser, tmp_file):
        data = {"mapping": {"n": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hi"]}, "create_time": 1}}}}
        sessions = parser.parse_file(tmp_file(json.dumps(data)), fmt="auto")
        assert len(sessions) >= 1


# -- Claude format ---------------------------------------------------------

class TestClaude:
    def test_claude_list_format(self, parser, tmp_file):
        data = [
            {"sender": "human", "text": "What is AI?"},
            {"sender": "assistant", "text": "AI stands for artificial intelligence."},
        ]
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        turns = sessions[0].turns
        assert sessions[0].session_id == "input"
        assert len(turns) == 2
        assert turns[0].role.value == "user"
        assert turns[1].role.value == "assistant"

    def test_claude_chat_messages_wrapper(self, parser, tmp_file):
        data = {
            "chat_messages": [
                {"sender": "human", "text": "Hello"},
                {"sender": "assistant", "text": "Hi there"},
            ]
        }
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert len(sessions[0].turns) == 2

    def test_claude_content_as_list(self, parser, tmp_file):
        data = [
            {"sender": "human", "content": [{"text": "Part 1"}, {"text": "Part 2"}]},
            {"sender": "assistant", "text": "Response"},
        ]
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        assert "Part 1" in sessions[0].turns[0].content
        assert "Part 2" in sessions[0].turns[0].content

    def test_claude_skips_empty_content(self, parser, tmp_file):
        data = [
            {"sender": "human", "text": "Hello"},
            {"sender": "assistant", "text": ""},
            {"sender": "human", "text": "Anyone there?"},
        ]
        sessions = parser.parse_file(tmp_file(json.dumps(data)))
        turns = sessions[0].turns
        assert len(turns) == 2  # empty assistant turn skipped

    def test_claude_auto_detected(self, parser, tmp_file):
        data = [{"sender": "human", "text": "Hi"}, {"sender": "assistant", "text": "Hello"}]
        sessions = parser.parse_file(tmp_file(json.dumps(data)), fmt="auto")
        assert len(sessions) >= 1


# -- Plain text format -----------------------------------------------------

class TestPlainText:
    def test_basic_plain_text(self, parser, tmp_file):
        text = textwrap.dedent("""\
            User: What is Python?
            Assistant: Python is a programming language.
            User: Thanks!
            Assistant: You're welcome.
        """)
        sessions = parser.parse_file(tmp_file(text, name="chat.txt"))
        turns = sessions[0].turns
        assert sessions[0].session_id == "chat"
        assert len(turns) == 4
        assert turns[0].role.value == "user"
        assert turns[1].role.value == "assistant"

    def test_case_insensitive_roles(self, parser, tmp_file):
        text = "human: Hello\nAI: Hi there\nUSER: How are you?\nassistant: Good"
        sessions = parser.parse_file(tmp_file(text, name="chat.txt"))
        assert len(sessions[0].turns) == 4

    def test_multiline_content(self, parser, tmp_file):
        text = textwrap.dedent("""\
            User: Tell me about Python.
            I want to know the basics.
            Assistant: Python is a language.
            It is used for many things.
        """)
        sessions = parser.parse_file(tmp_file(text, name="chat.txt"))
        turns = sessions[0].turns
        assert len(turns) == 2
        assert "basics" in turns[0].content
        assert "many things" in turns[1].content

    def test_plain_text_auto_detected(self, parser, tmp_file):
        text = "User: Hello\nAssistant: Hi"
        sessions = parser.parse_file(tmp_file(text, name="chat.txt"), fmt="auto")
        assert len(sessions) >= 1


# -- Flat file semantics --------------------------------------------------

class TestFlatFileSemantics:
    def test_flat_export_stays_single_session(self, tmp_file):
        parser = ConversationParser()
        data = [
            {"sender": "human", "text": f"Message {i}"}
            if i % 2 == 0
            else {"sender": "assistant", "text": f"Reply {i}"}
            for i in range(10)
        ]
        sessions = parser.parse_file(tmp_file(json.dumps(data), name="thread.json"))
        assert len(sessions) == 1
        assert sessions[0].session_id == "thread"
        assert len(sessions[0].turns) == 10


# -- Format detection ------------------------------------------------------

class TestFormatDetection:
    def test_detects_native_json(self, parser):
        raw = json.dumps({"sessions": [{"session_id": "s1", "turns": []}]})
        assert parser._detect_format(raw) == "json"

    def test_detects_chatgpt(self, parser):
        raw = json.dumps({"mapping": {}})
        assert parser._detect_format(raw) == "chatgpt"

    def test_detects_claude_list(self, parser):
        raw = json.dumps([{"sender": "human", "text": "hi"}])
        assert parser._detect_format(raw) == "claude"

    def test_detects_claude_wrapper(self, parser):
        raw = json.dumps({"chat_messages": [{"sender": "human", "text": "hi"}]})
        assert parser._detect_format(raw) == "claude"

    def test_detects_plain_text(self, parser):
        assert parser._detect_format("User: hello\nAssistant: hi") == "plain"

    def test_falls_back_to_plain(self, parser):
        assert parser._detect_format("some random text") == "plain"


# -- Error handling --------------------------------------------------------

class TestErrors:
    def test_file_not_found(self, parser):
        with pytest.raises(FileNotFoundError):
            parser.parse_file("/nonexistent/path.json")

    def test_empty_file(self, parser, tmp_file):
        with pytest.raises(ValueError, match="empty"):
            parser.parse_file(tmp_file(""))

    def test_no_turns_in_plain_text(self, parser, tmp_file):
        with pytest.raises(ValueError, match="No turns"):
            parser.parse_file(tmp_file("just some random text\nno roles here", name="chat.txt"), fmt="plain")

    def test_no_messages_in_chatgpt(self, parser, tmp_file):
        with pytest.raises(ValueError, match="No user/assistant"):
            parser.parse_file(tmp_file(json.dumps({"mapping": {}})), fmt="chatgpt")

    def test_no_messages_in_claude(self, parser, tmp_file):
        with pytest.raises(ValueError, match="No human/assistant"):
            parser.parse_file(tmp_file(json.dumps([])), fmt="claude")

    def test_malformed_json_raises_clear_error(self, parser, tmp_file):
        with pytest.raises(ValueError, match="Malformed JSON input"):
            parser.parse_file(tmp_file("{not valid json"))


# -- TranscriptLoader integration -----------------------------------------

class TestTranscriptLoader:
    def test_loader_uses_parser_for_chatgpt(self, tmp_file):
        from dormancy_detect.transcript_loader import TranscriptLoader
        data = {
            "mapping": {
                "n1": {"message": {"author": {"role": "user"}, "content": {"parts": ["Hi"]}, "create_time": 1}},
                "n2": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Hello"]}, "create_time": 2}},
            }
        }
        loader = TranscriptLoader()
        transcript = loader.load(tmp_file(json.dumps(data)))
        assert len(transcript.sessions) >= 1
        assert transcript.sessions[0].turns[0].content == "Hi"
        assert transcript.sessions[0].session_id == "input"

    def test_loader_uses_parser_for_claude(self, tmp_file):
        from dormancy_detect.transcript_loader import TranscriptLoader
        data = [
            {"sender": "human", "text": "Hello"},
            {"sender": "assistant", "text": "Hi"},
        ]
        loader = TranscriptLoader()
        transcript = loader.load(tmp_file(json.dumps(data)))
        assert len(transcript.sessions) >= 1

    def test_loader_uses_parser_for_plain_text(self, tmp_file):
        from dormancy_detect.transcript_loader import TranscriptLoader
        text = "User: Hello\nAssistant: Hi"
        loader = TranscriptLoader()
        transcript = loader.load(tmp_file(text, name="chat.txt"))
        assert len(transcript.sessions) >= 1
        assert transcript.sessions[0].session_id == "chat"

    def test_loader_explicit_format(self, tmp_file):
        from dormancy_detect.transcript_loader import TranscriptLoader
        data = [
            {"sender": "human", "text": "Hello"},
            {"sender": "assistant", "text": "Hi"},
        ]
        loader = TranscriptLoader(fmt="claude")
        transcript = loader.load(tmp_file(json.dumps(data)))
        assert len(transcript.sessions) >= 1

    def test_loader_native_json_still_works(self, tmp_file):
        from dormancy_detect.transcript_loader import TranscriptLoader
        data = {
            "sessions": [
                {
                    "session_id": "s1",
                    "turns": [
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi"},
                    ],
                }
            ]
        }
        loader = TranscriptLoader()
        transcript = loader.load(tmp_file(json.dumps(data)))
        assert len(transcript.sessions) == 1
        assert transcript.sessions[0].session_id == "s1"

    def test_loader_directory_ignores_markdown(self, tmp_file, tmp_path: Path):
        from dormancy_detect.transcript_loader import TranscriptLoader
        text = "User: Hello\nAssistant: Hi"
        tmp_file(text, name="session1.txt")
        tmp_file("# Notes", name="README.md")
        loader = TranscriptLoader()
        transcript = loader.load(tmp_path)
        assert len(transcript.sessions) == 1
