import pytest


def test_arithmetic_operations():
    """Test basic arithmetic operations."""
    assert 1 + 1 == 2
    assert 5 - 3 == 2
    assert 4 * 3 == 12
    assert 10 / 2 == 5


def test_string_operations():
    """Test string manipulation and comparison."""
    assert 'hello'.upper() == 'HELLO'
    assert 'WORLD'.lower() == 'world'
    assert 'python' in 'I love python'
    assert 'test'.capitalize() == 'Test'


def test_list_operations():
    """Test list operations and properties."""
    assert len([1, 2, 3]) == 3
    test_list = [1, 2, 3, 4, 5]
    assert test_list[0] == 1
    assert test_list[-1] == 5
    assert sum([1, 2, 3]) == 6


def test_boolean_logic():
    """Test boolean operations and comparisons."""
    assert True is True
    assert False is False
    assert not False
    assert True and True
    assert True or False


def test_data_types():
    """Test basic data type checks."""
    assert isinstance(42, int)
    assert isinstance('hello', str)
    assert isinstance([1, 2, 3], list)
    assert isinstance({'key': 'value'}, dict)
    assert isinstance(3.14, float)