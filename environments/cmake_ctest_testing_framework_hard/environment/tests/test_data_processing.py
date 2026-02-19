#!/usr/bin/env python3
"""Test suite for data processing library."""

import pytest


@pytest.fixture
def sample_data():
    """Provide sample test data for tests."""
    return {
        'numbers': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'strings': ['hello', 'world', 'test', 'data'],
        'dict_data': {'name': 'Alice', 'age': 30, 'city': 'NYC'}
    }


def test_filter_even_numbers(sample_data):
    """Test filtering even numbers from a list."""
    numbers = sample_data['numbers']
    evens = [n for n in numbers if n % 2 == 0]
    assert len(evens) == 5
    assert all(n % 2 == 0 for n in evens)


def test_list_mapping():
    """Test mapping operations on lists."""
    numbers = [1, 2, 3, 4, 5]
    squared = [n * n for n in numbers]
    assert squared == [1, 4, 9, 16, 25]
    assert len(squared) == len(numbers)


def test_dictionary_operations(sample_data):
    """Test dictionary key existence and value updates."""
    data = sample_data['dict_data'].copy()
    assert 'name' in data
    assert data['name'] == 'Alice'
    data['age'] = 31
    assert data['age'] == 31
    assert len(data) == 3


def test_string_processing(sample_data):
    """Test string splitting and joining operations."""
    strings = sample_data['strings']
    joined = ' '.join(strings)
    assert joined == 'hello world test data'
    split_result = joined.split(' ')
    assert split_result == strings


def test_numeric_calculations():
    """Test basic numeric operations like sum and average."""
    numbers = [10, 20, 30, 40, 50]
    total = sum(numbers)
    assert total == 150
    average = total / len(numbers)
    assert average == 30.0
    assert min(numbers) == 10
    assert max(numbers) == 50


def test_data_validation(sample_data):
    """Test data validation and type checking."""
    numbers = sample_data['numbers']
    assert isinstance(numbers, list)
    assert all(isinstance(n, int) for n in numbers)
    assert len(numbers) > 0
    assert min(numbers) > 0