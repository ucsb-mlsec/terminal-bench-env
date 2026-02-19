import pytest
import os
from user_manager import UserManager


@pytest.fixture(scope="module")
def db_path():
    return '/tmp/test_users.db'


@pytest.fixture(scope="module")
def user_manager(db_path):
    manager = UserManager(db_path)
    manager.create_table()
    yield manager