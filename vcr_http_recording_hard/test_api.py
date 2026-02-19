#!/usr/bin/env python
import pytest
import vcr
import requests
import json

# VCR configuration with intentional issues
my_vcr = vcr.VCR(
    cassette_library_dir='cassettes',
    record_mode='once',
    match_on=['uri', 'method'],
    path_transformer=vcr.VCR.ensure_suffix('.yaml')
)

@my_vcr.use_cassette('test_weather')
def test_weather():
    """Test weather API endpoint"""
    response = requests.get('https://api.example.com/weather?city=London')
    assert response.status_code == 200
    data = response.json()
    assert 'temperature' in data

@my_vcr.use_cassette('test_users')
def test_users():
    """Test users API endpoint"""
    response = requests.get('https://api.example.com/users/1')
    assert response.status_code == 200
    data = response.json()
    assert 'name' in data

@my_vcr.use_cassette('test_posts')
def test_posts():
    """Test posts API endpoint"""
    response = requests.get('https://api.example.com/posts/1')
    assert response.status_code == 200
    data = response.json()
    assert 'title' in data