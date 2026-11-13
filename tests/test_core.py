import unittest
from unittest.mock import patch
from src.core.message import Message, MessageRole
from src.core.events import StreamChunk, ResponseMetadata
from src.core.security import CredentialStore


class TestCoreDataModels(unittest.TestCase):
    def test_message_creation_and_roles(self):
        msg_user = Message(role=MessageRole.USER, content="Hello")
        self.assertEqual(msg_user.role, MessageRole.USER)
        self.assertEqual(msg_user.content, "Hello")

        # Test string role conversion
        msg_str = Message(role="assistant", content="Hi")
        self.assertEqual(msg_str.role, MessageRole.ASSISTANT)

        msg_sys = Message(role="system", content="You are a helpful assistant")
        self.assertEqual(msg_sys.role, MessageRole.SYSTEM)

    def test_message_dict_compatibility(self):
        msg = Message(role=MessageRole.USER, content="Test")
        # Dict access compatibility
        self.assertEqual(msg["role"], "user")
        self.assertEqual(msg["content"], "Test")
        self.assertEqual(msg.get("content"), "Test")
        self.assertEqual(msg.to_dict(), {"role": "user", "content": "Test"})

        # From dict
        reconstructed = Message.from_dict({"role": "assistant", "content": "Reply"})
        self.assertEqual(reconstructed.role, MessageRole.ASSISTANT)
        self.assertEqual(reconstructed.content, "Reply")

    def test_message_equality(self):
        msg1 = Message("user", "Same")
        msg2 = Message(MessageRole.USER, "Same")
        self.assertEqual(msg1, msg2)
        self.assertEqual(msg1, {"role": "user", "content": "Same"})

    def test_stream_chunk_and_metadata(self):
        meta = ResponseMetadata(
            model="llama3.2:3b",
            provider_id="ollama",
            total_tokens=42,
            finish_reason="stop",
        )
        chunk = StreamChunk(delta="Hello ", is_final=False, metadata=meta)
        self.assertEqual(chunk.delta, "Hello ")
        self.assertEqual(chunk.content, "Hello ")
        self.assertFalse(chunk.is_final)
        self.assertEqual(chunk.metadata.model, "llama3.2:3b")

    def test_credential_store(self):
        store = CredentialStore("RelayAITest")
        with patch("keyring.set_password") as mock_set, \
             patch("keyring.get_password", return_value="secret_key_123") as mock_get, \
             patch("keyring.delete_password") as mock_del:

            res_set = store.set_api_key("gemini", "secret_key_123")
            self.assertTrue(res_set)
            mock_set.assert_called_once_with("RelayAITest", "gemini", "secret_key_123")

            key = store.get_api_key("gemini")
            self.assertEqual(key, "secret_key_123")

            self.assertTrue(store.has_api_key("gemini"))

            res_del = store.delete_api_key("gemini")
            self.assertTrue(res_del)
            mock_del.assert_called_once_with("RelayAITest", "gemini")


if __name__ == "__main__":
    unittest.main()

