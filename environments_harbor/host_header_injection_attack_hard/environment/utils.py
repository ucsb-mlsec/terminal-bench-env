#!/usr/bin/env python3

import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import random
import string
import datetime
import sqlite3
import os

# Configure logging
logging.basicConfig(
    filename='/var/log/webapp/error.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def generate_token(length=32):
    """
    Generate a random alphanumeric token.
    
    Args:
        length (int): Length of the token to generate. Default is 32.
    
    Returns:
        str: Random alphanumeric token
    """
    characters = string.ascii_letters + string.digits
    token = ''.join(random.choice(characters) for _ in range(length))
    logger.info(f"Generated token of length {length}")
    return token


def send_reset_email(to_email, reset_link, username):
    """
    Send password reset email to the specified email address.
    
    Args:
        to_email (str): Recipient email address
        reset_link (str): Password reset link to include in email
        username (str): Username of the account requesting reset
    
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        logger.info(f"Attempting to send password reset email to {to_email} for user {username}")
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = 'Password Reset Request'
        msg['From'] = 'noreply@webapp.local'
        msg['To'] = to_email
        
        # Email body with reset link
        text_body = f"""
Hello {username},

You have requested a password reset for your account.

Please click the following link to reset your password:
{reset_link}

If you did not request this password reset, please ignore this email.

This link will expire in 24 hours.

Best regards,
WebApp Support Team
"""
        
        html_body = f"""
<html>
<head></head>
<body>
    <p>Hello {username},</p>
    <p>You have requested a password reset for your account.</p>
    <p>Please click the following link to reset your password:</p>
    <p><a href="{reset_link}">{reset_link}</a></p>
    <p>If you did not request this password reset, please ignore this email.</p>
    <p>This link will expire in 24 hours.</p>
    <p>Best regards,<br>WebApp Support Team</p>
</body>
</html>
"""
        
        part1 = MIMEText(text_body, 'plain')
        part2 = MIMEText(html_body, 'html')
        
        msg.attach(part1)
        msg.attach(part2)
        
        # Mock SMTP connection (logging only for security testing)
        logger.info(f"SMTP: Connecting to mail server for {to_email}")
        logger.info(f"SMTP: Sending email with reset link: {reset_link}")
        logger.info(f"SMTP: Email body contains: {text_body[:100]}...")
        
        # In production, this would be actual SMTP:
        # with smtplib.SMTP('localhost', 25) as server:
        #     server.send_message(msg)
        
        logger.info(f"Successfully sent password reset email to {to_email} for user {username}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send reset email to {to_email}: {str(e)}")
        return False


def log_reset_attempt(username, ip, host_header):
    """
    Log password reset attempt with details.
    
    Args:
        username (str): Username requesting password reset
        ip (str): IP address of the requester
        host_header (str): Host header from the request
    """
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logger.info(f"Password reset attempt - User: {username}, IP: {ip}, Host: {host_header}, Time: {timestamp}")


def store_reset_token(username, token, reset_link, ip):
    """
    Store password reset token information in database.
    
    Args:
        username (str): Username associated with the token
        token (str): Reset token generated
        reset_link (str): Complete reset link with token
        ip (str): IP address of the requester
    
    Returns:
        bool: True if stored successfully, False otherwise
    """
    try:
        db_path = '/var/backups/sessions.db'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reset_tokens (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                token TEXT NOT NULL,
                reset_link TEXT NOT NULL,
                ip_address TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                used INTEGER DEFAULT 0
            )
        ''')
        
        # Insert token information
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO reset_tokens (username, token, reset_link, ip_address, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, token, reset_link, ip, timestamp))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Stored reset token for user {username} from IP {ip}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to store reset token for {username}: {str(e)}")
        return False


if __name__ == '__main__':
    # Test functions
    print("Utils module loaded successfully")
    token = generate_token()
    print(f"Sample token: {token}")