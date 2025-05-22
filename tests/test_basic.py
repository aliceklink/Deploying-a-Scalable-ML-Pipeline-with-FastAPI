"""Basic tests to ensure GitHub Actions work."""

def test_basic_assertion():
    """Test that basic assertions work."""
    assert 1 + 1 == 2

def test_string_operations():
    """Test basic string operations."""
    test_string = "hello world"
    assert test_string.split() == ["hello", "world"]

def test_list_operations():
    """Test basic list operations."""
    test_list = [1, 2, 3, 4, 5]
    assert len(test_list) == 5
    assert sum(test_list) == 15
