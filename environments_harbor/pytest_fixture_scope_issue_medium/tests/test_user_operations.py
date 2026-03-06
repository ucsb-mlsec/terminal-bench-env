import pytest
import sqlite3


def test_add_user(user_manager):
    user_manager.add_user(username='john', email='john@example.com')
    user = user_manager.get_user('john')
    assert user is not None
    assert user['username'] == 'john'
    assert user['email'] == 'john@example.com'


def test_add_duplicate_user(user_manager):
    user_manager.add_user(username='jane', email='jane@example.com')
    with pytest.raises(sqlite3.IntegrityError):
        user_manager.add_user(username='jane', email='jane2@example.com')


def test_get_all_users(user_manager):
    user_manager.add_user(username='alice', email='alice@example.com')
    user_manager.add_user(username='bob', email='bob@example.com')
    users = user_manager.get_all_users()
    assert len(users) >= 2


def test_delete_user(user_manager):
    user_manager.add_user(username='charlie', email='charlie@example.com')
    user_manager.delete_user('charlie')
    user = user_manager.get_user('charlie')
    assert user is None