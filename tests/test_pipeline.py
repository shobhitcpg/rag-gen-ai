import pytest
from unittest.mock import MagicMock, patch
from src.retrieval import RetrievalEngine

@pytest.fixture
def mock_engine():
    mock_embedder = MagicMock()
    mock_vector_store = MagicMock()
    engine = RetrievalEngine(mock_embedder, mock_vector_store)
    return engine
def test_strategy_a(mock_engine):
    mock_engine.query = MagicMock(return_value=[("Doc1", 0.9), ("Doc2", 0.8)])
    result = mock_engine.strategy_a("Test query")
    assert "Doc1" in result and "Doc2" in result
def test_strategy_b_with_mock(mock_engine):
    mock_engine.use_mock = True
    mock_engine.mock_query_expansion = MagicMock(return_value="Expanded mock query")
    mock_engine.query = MagicMock(return_value=[("Doc3", 0.85), ("Doc4", 0.75)])
    
    result = mock_engine.strategy_b("Test query")
    
    mock_engine.mock_query_expansion.assert_called_once_with("Test query")
    mock_engine.query.assert_called_once_with("Expanded mock query", top_k=5)
    assert "Doc3" in result and "Doc4" in result

@patch("src.retrieval.RetrievalEngine.vertex_query_expansion")
def test_strategy_b_with_vertexai(mock_vertex_expansion, mock_engine):
    mock_engine.use_mock = False
    mock_vertex_expansion.return_value = "Expanded vertex query"
    mock_engine.query = MagicMock(return_value=[("Doc5", 0.95), ("Doc6", 0.85)])
    
    result = mock_engine.strategy_b("Test query")
    
    mock_vertex_expansion.assert_called_once_with("Test query")
    mock_engine.query.assert_called_once_with("Expanded vertex query", top_k=5)
    assert "Doc5" in result and "Doc6" in result
