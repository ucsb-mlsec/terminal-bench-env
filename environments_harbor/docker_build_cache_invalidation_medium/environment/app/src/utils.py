import re
import uuid
from datetime import datetime


def format_timestamp(timestamp=None, format_string="%Y-%m-%d %H:%M:%S"):
    """
    Format a timestamp to a readable string format.
    If no timestamp provided, uses current time.
    """
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime(format_string)


def validate_email(email):
    """
    Validate an email address using regex pattern.
    Returns True if valid, False otherwise.
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def generate_random_id(prefix=""):
    """
    Generate a random UUID-based identifier.
    Optionally prepend a prefix to the ID.
    """
    random_id = str(uuid.uuid4())
    return f"{prefix}{random_id}" if prefix else random_id


def sanitize_string(input_string):
    """
    Remove special characters and extra whitespace from string.
    Returns cleaned string with single spaces.
    """
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    return ' '.join(cleaned.split())