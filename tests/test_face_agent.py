"""Tests for face_agent and face_tools modules.

All InsightFace and Firestore interactions are mocked.
"""

import base64
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure SightLine root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embedding(seed: int = 0) -> np.ndarray:
    """Generate a deterministic 512-D L2-normalized embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_mock_face(seed: int = 0, det_score: float = 0.99):
    """Create a mock InsightFace face object."""
    face = MagicMock()
    face.bbox = np.array([10, 20, 110, 120], dtype=np.float32)
    face.det_score = det_score
    face.normed_embedding = _make_embedding(seed)
    return face


def _minimal_jpeg_bytes() -> bytes:
    """Return minimal valid JPEG bytes for testing."""
    # 1x1 white JPEG
    return base64.b64decode(
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkS"
        "Ew8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJ"
        "CQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIy"
        "MjIyMjIyMjIyMjIyMjL/wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEA"
        "AAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIh"
        "MUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6"
        "Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZ"
        "mqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx"
        "8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREA"
        "AgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAV"
        "YnLRChYkNOEl8RcYI4Q/RFhHRUYnJCk2NTg5OkNERUZHSElKU1RVVldYWVpjZGVm"
        "Z2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6"
        "wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEA"
        "PwD3+gD/2Q=="
    )


def _minimal_jpeg_b64() -> str:
    """Return base64-encoded minimal JPEG for testing."""
    return base64.b64encode(_minimal_jpeg_bytes()).decode()


# ---------------------------------------------------------------------------
# face_agent tests
# ---------------------------------------------------------------------------


class TestDetectFaces:
    """Tests for face_agent.detect_faces."""

    @patch("agents.face_agent._get_face_app")
    def test_detect_returns_faces(self, mock_get_app):
        from agents.face_agent import detect_faces

        mock_face = _make_mock_face(seed=1)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        results = detect_faces(_minimal_jpeg_bytes())

        assert len(results) == 1
        assert results[0]["score"] == pytest.approx(0.99)
        assert len(results[0]["bbox"]) == 4
        np.testing.assert_array_equal(
            results[0]["embedding"], mock_face.normed_embedding
        )

    @patch("agents.face_agent._get_face_app")
    def test_detect_no_faces(self, mock_get_app):
        from agents.face_agent import detect_faces

        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_get_app.return_value = mock_app

        results = detect_faces(_minimal_jpeg_bytes())
        assert results == []

    def test_detect_invalid_image(self):
        from agents.face_agent import detect_faces

        with pytest.raises(ValueError, match="Failed to decode"):
            detect_faces(b"not-an-image")


class TestGenerateEmbedding:
    """Tests for face_agent.generate_embedding."""

    @patch("agents.face_agent._get_face_app")
    def test_returns_512d_embedding(self, mock_get_app):
        from agents.face_agent import generate_embedding

        mock_face = _make_mock_face(seed=2)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        emb = generate_embedding(dummy_img)

        assert emb.shape == (512,)
        # Verify L2 normalization
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    @patch("agents.face_agent._get_face_app")
    def test_no_face_raises(self, mock_get_app):
        from agents.face_agent import generate_embedding

        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_get_app.return_value = mock_app

        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="No face detected"):
            generate_embedding(dummy_img)


class TestMatchFace:
    """Tests for face_agent.match_face."""

    def test_match_above_threshold(self):
        from agents.face_agent import match_face

        emb = _make_embedding(seed=10)
        library = [
            {
                "face_id": "f1",
                "person_name": "Alice",
                "relationship": "friend",
                "embedding": emb,  # exact same embedding -> similarity 1.0
            },
        ]
        result = match_face(emb, library)
        assert result is not None
        assert result["person_name"] == "Alice"
        assert result["similarity"] == pytest.approx(1.0, abs=1e-5)
        assert "embedding" not in result  # should be stripped

    def test_no_match_below_threshold(self):
        from agents.face_agent import match_face

        emb_query = _make_embedding(seed=20)
        emb_lib = _make_embedding(seed=99)  # very different
        library = [
            {
                "face_id": "f2",
                "person_name": "Bob",
                "relationship": "colleague",
                "embedding": emb_lib,
            },
        ]
        result = match_face(emb_query, library)
        # With random seeds the cosine sim should be near 0
        assert result is None

    def test_empty_library(self):
        from agents.face_agent import match_face

        emb = _make_embedding(seed=30)
        assert match_face(emb, []) is None

    def test_best_match_selected(self):
        from agents.face_agent import match_face

        emb = _make_embedding(seed=40)
        # Create a slightly perturbed version (high similarity)
        emb_close = emb.copy()
        emb_close[:5] += 0.01
        emb_close /= np.linalg.norm(emb_close)

        library = [
            {
                "face_id": "f3",
                "person_name": "Far",
                "relationship": "stranger",
                "embedding": _make_embedding(seed=99),
            },
            {
                "face_id": "f4",
                "person_name": "Close",
                "relationship": "friend",
                "embedding": emb_close,
            },
        ]
        result = match_face(emb, library)
        assert result is not None
        assert result["person_name"] == "Close"


class TestIdentifyPersonsInFrame:
    """Tests for face_agent.identify_persons_in_frame."""

    @patch("agents.face_agent._get_face_app")
    def test_identify_known_person(self, mock_get_app):
        from agents.face_agent import identify_persons_in_frame

        emb = _make_embedding(seed=50)
        mock_face = _make_mock_face(seed=50)  # same seed -> same embedding
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        library = [
            {
                "face_id": "f10",
                "person_name": "Charlie",
                "relationship": "brother",
                "embedding": emb,
            },
        ]

        results = identify_persons_in_frame(
            _minimal_jpeg_b64(), "user_001", face_library=library
        )

        assert len(results) == 1
        assert results[0]["person_name"] == "Charlie"
        assert results[0]["relationship"] == "brother"
        assert results[0]["similarity"] > 0.4

    @patch("agents.face_agent._get_face_app")
    def test_identify_unknown_person(self, mock_get_app):
        from agents.face_agent import identify_persons_in_frame

        mock_face = _make_mock_face(seed=60)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        # Library with a different embedding
        library = [
            {
                "face_id": "f11",
                "person_name": "Diana",
                "relationship": "friend",
                "embedding": _make_embedding(seed=99),
            },
        ]

        results = identify_persons_in_frame(
            _minimal_jpeg_b64(), "user_001", face_library=library
        )

        assert len(results) == 1
        assert results[0]["person_name"] == "unknown"
        assert results[0]["similarity"] == 0.0

    @patch("agents.face_agent._get_face_app")
    def test_identify_empty_library(self, mock_get_app):
        from agents.face_agent import identify_persons_in_frame

        mock_face = _make_mock_face(seed=70)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        results = identify_persons_in_frame(
            _minimal_jpeg_b64(), "user_001", face_library=[]
        )

        assert len(results) == 1
        assert results[0]["person_name"] == "unknown"

    @patch("agents.face_agent._get_face_app")
    def test_identify_no_faces(self, mock_get_app):
        from agents.face_agent import identify_persons_in_frame

        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_get_app.return_value = mock_app

        results = identify_persons_in_frame(
            _minimal_jpeg_b64(), "user_001", face_library=[]
        )
        assert results == []


# ---------------------------------------------------------------------------
# face_tools tests
# ---------------------------------------------------------------------------


class MockDocRef:
    """Minimal mock Firestore document reference."""

    def __init__(self, doc_id: str, data: dict | None = None):
        self.id = doc_id
        self._data = data
        self.reference = self

    def set(self, data: dict):
        self._data = data

    def get(self):
        snap = MagicMock()
        snap.exists = self._data is not None
        snap.to_dict.return_value = self._data
        snap.id = self.id
        return snap

    def delete(self):
        self._data = None

    def to_dict(self):
        return self._data


class MockCollection:
    """Minimal mock Firestore collection."""

    def __init__(self, docs: list[MockDocRef] | None = None):
        self._docs = docs or []
        self._auto_id_counter = 0

    def document(self, doc_id: str | None = None):
        if doc_id is None:
            self._auto_id_counter += 1
            new_doc = MockDocRef(f"auto_{self._auto_id_counter}")
            self._docs.append(new_doc)
            return new_doc
        for doc in self._docs:
            if doc.id == doc_id:
                return doc
        new_doc = MockDocRef(doc_id)
        self._docs.append(new_doc)
        return new_doc

    def where(self, field, op, value):
        filtered = [
            d for d in self._docs
            if d._data and d._data.get(field) == value
        ]
        mock_query = MagicMock()
        mock_query.stream.return_value = [d.get() for d in filtered]
        # Make delete work on the returned docs
        for snap in mock_query.stream.return_value:
            snap.reference = MagicMock()
            snap.reference.delete = MagicMock()
        mock_query.stream.return_value = iter(
            [_make_doc_snapshot(d) for d in filtered]
        )
        return mock_query

    def stream(self):
        return iter([_make_doc_snapshot(d) for d in self._docs if d._data])


def _make_doc_snapshot(doc_ref: MockDocRef):
    """Create a mock DocumentSnapshot from a MockDocRef."""
    snap = MagicMock()
    snap.id = doc_ref.id
    snap.exists = doc_ref._data is not None
    snap.to_dict.return_value = doc_ref._data
    snap.reference = doc_ref
    return snap


class TestFaceToolsRegister:
    """Tests for face_tools.register_face."""

    @patch("agents.face_agent._get_face_app")
    def test_register_face_success(self, mock_get_app):
        from tools.face_tools import register_face, set_db_client

        mock_face = _make_mock_face(seed=100)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        mock_coll = MockCollection()
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        result = register_face(
            user_id="u1",
            person_name="Eve",
            relationship="sister",
            image_base64=_minimal_jpeg_b64(),
            photo_index=0,
        )

        assert result["person_name"] == "Eve"
        assert result["relationship"] == "sister"
        assert result["photo_index"] == 0
        assert result["stored_reference_photo"] is False
        assert result["consent_confirmed"] is False
        assert "face_id" in result
        assert "created_at" in result

        # Cleanup
        set_db_client(None)

    @patch("agents.face_agent._get_face_app")
    def test_register_face_with_consent_and_reference_photo(self, mock_get_app):
        from tools.face_tools import register_face, set_db_client

        mock_face = _make_mock_face(seed=101)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        mock_coll = MockCollection()
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        result = register_face(
            user_id="u1",
            person_name="Grace",
            relationship="friend",
            image_base64=_minimal_jpeg_b64(),
            photo_index=1,
            consent_confirmed=True,
            store_reference_photo=True,
        )

        assert result["person_name"] == "Grace"
        assert result["stored_reference_photo"] is True
        assert result["consent_confirmed"] is True

        saved = mock_coll._docs[0]._data
        assert saved["consent_confirmed"] is True
        assert "consent_timestamp" in saved
        assert "reference_photo_base64" in saved
        assert "reference_photo_sha256" in saved
        assert saved["reference_photo_bytes"] > 0

        set_db_client(None)

    @patch("agents.face_agent._get_face_app")
    def test_register_face_store_photo_requires_consent(self, mock_get_app):
        from tools.face_tools import register_face, set_db_client

        mock_face = _make_mock_face(seed=102)
        mock_app = MagicMock()
        mock_app.get.return_value = [mock_face]
        mock_get_app.return_value = mock_app

        mock_coll = MockCollection()
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        with pytest.raises(ValueError, match="Consent is required"):
            register_face(
                user_id="u1",
                person_name="Hank",
                relationship="friend",
                image_base64=_minimal_jpeg_b64(),
                store_reference_photo=True,
            )

        set_db_client(None)

    @patch("agents.face_agent._get_face_app")
    def test_register_face_no_face_detected(self, mock_get_app):
        from tools.face_tools import register_face, set_db_client

        mock_app = MagicMock()
        mock_app.get.return_value = []
        mock_get_app.return_value = mock_app

        mock_db = MagicMock()
        set_db_client(mock_db)

        with pytest.raises(ValueError, match="No face detected"):
            register_face(
                user_id="u1",
                person_name="Nobody",
                relationship="unknown",
                image_base64=_minimal_jpeg_b64(),
            )

        set_db_client(None)


class TestFaceToolsDelete:
    """Tests for face_tools delete operations."""

    def test_delete_face_exists(self):
        from tools.face_tools import delete_face, set_db_client

        doc = MockDocRef("face_1", {"person_name": "Alice"})
        mock_coll = MockCollection([doc])
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        assert delete_face("u1", "face_1") is True

        set_db_client(None)

    def test_delete_face_not_found(self):
        from tools.face_tools import delete_face, set_db_client

        mock_coll = MockCollection()
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        assert delete_face("u1", "nonexistent") is False

        set_db_client(None)

    def test_delete_all_faces(self):
        from tools.face_tools import delete_all_faces, set_db_client

        docs = [
            MockDocRef("f1", {"person_name": "Bob", "embedding": [0.1] * 512}),
            MockDocRef("f2", {"person_name": "Bob", "embedding": [0.2] * 512}),
            MockDocRef("f3", {"person_name": "Carol", "embedding": [0.3] * 512}),
        ]
        mock_coll = MockCollection(docs)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        count = delete_all_faces("u1", "Bob")
        assert count == 2

        set_db_client(None)


class TestFaceToolsList:
    """Tests for face_tools.list_faces."""

    def test_list_faces(self):
        from tools.face_tools import list_faces, set_db_client

        docs = [
            MockDocRef("f1", {
                "person_name": "Alice",
                "relationship": "friend",
                "photo_index": 0,
                "consent_confirmed": True,
                "reference_photo_base64": "abc123",
                "created_at": "2025-01-01T00:00:00",
            }),
            MockDocRef("f2", {
                "person_name": "Bob",
                "relationship": "colleague",
                "photo_index": 1,
                "consent_confirmed": False,
                "created_at": "2025-01-02T00:00:00",
            }),
        ]
        mock_coll = MockCollection(docs)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        result = list_faces("u1")
        assert len(result) == 2
        assert result[0]["person_name"] == "Alice"
        assert result[1]["person_name"] == "Bob"
        assert result[0]["consent_confirmed"] is True
        assert result[0]["has_reference_photo"] is True
        assert result[1]["consent_confirmed"] is False
        assert result[1]["has_reference_photo"] is False
        # Embeddings should NOT be in the list output
        assert "embedding" not in result[0]

        set_db_client(None)

    def test_list_faces_empty(self):
        from tools.face_tools import list_faces, set_db_client

        mock_coll = MockCollection()
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        result = list_faces("u1")
        assert result == []

        set_db_client(None)


class TestFaceToolsLoad:
    """Tests for face_tools.load_face_library."""

    def test_load_library(self):
        from tools.face_tools import load_face_library, set_db_client

        class _MockVector:
            """Mimics Firestore Vector with only .value (no to_map_value)."""
            def __init__(self, val):
                self.value = val
        emb_vec = _MockVector(_make_embedding(seed=200).tolist())

        docs = [
            MockDocRef("f1", {
                "person_name": "Alice",
                "relationship": "friend",
                "embedding": emb_vec,
            }),
        ]
        mock_coll = MockCollection(docs)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        library = load_face_library("u1")
        assert len(library) == 1
        assert library[0]["person_name"] == "Alice"
        assert library[0]["embedding"].shape == (512,)
        assert library[0]["embedding"].dtype == np.float32

        set_db_client(None)

    def test_load_library_skips_missing_embeddings(self):
        from tools.face_tools import load_face_library, set_db_client

        docs = [
            MockDocRef("f1", {
                "person_name": "NoEmb",
                "relationship": "unknown",
                "embedding": None,
            }),
        ]
        mock_coll = MockCollection(docs)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        library = load_face_library("u1")
        assert len(library) == 0

        set_db_client(None)

    def test_load_library_handles_list_embeddings(self):
        from tools.face_tools import load_face_library, set_db_client

        docs = [
            MockDocRef("f1", {
                "person_name": "ListEmb",
                "relationship": "friend",
                "embedding": _make_embedding(seed=300).tolist(),
            }),
        ]
        mock_coll = MockCollection(docs)
        mock_db = MagicMock()
        mock_db.collection.return_value.document.return_value.collection.return_value = (
            mock_coll
        )
        set_db_client(mock_db)

        library = load_face_library("u1")
        assert len(library) == 1
        assert library[0]["embedding"].shape == (512,)

        set_db_client(None)
