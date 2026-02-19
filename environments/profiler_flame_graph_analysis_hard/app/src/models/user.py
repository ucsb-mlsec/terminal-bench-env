#!/usr/bin/env python3
"""
User model module.
Defines the User class for representing application users.
"""

from datetime import datetime
from typing import Dict, Optional, Any
import re


class User:
    """
    User model class representing an application user.
    """
    
    def __init__(
        self,
        id: Optional[int] = None,
        username: str = "",
        email: str = "",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        is_active: bool = True
    ):
        """
        Initialize a User instance.
        
        Args:
            id: Unique user identifier
            username: User's username
            email: User's email address
            created_at: Timestamp when user was created
            updated_at: Timestamp when user was last updated
            is_active: Whether the user account is active
        """
        self.id = id
        self.username = username
        self.email = email
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.is_active = is_active
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert User instance to dictionary representation.
        
        Returns:
            Dictionary containing user data
        """
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """
        Create User instance from dictionary.
        
        Args:
            data: Dictionary containing user data
            
        Returns:
            User instance
        """
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            id=data.get('id'),
            username=data.get('username', ''),
            email=data.get('email', ''),
            created_at=created_at,
            updated_at=updated_at,
            is_active=data.get('is_active', True)
        )
    
    def validate(self) -> bool:
        """
        Validate user data.
        
        Returns:
            True if user data is valid, False otherwise
        """
        if not self.username or len(self.username) < 3:
            return False
        
        if not self.email or not self._is_valid_email(self.email):
            return False
        
        return True
    
    def _is_valid_email(self, email: str) -> bool:
        """
        Check if email format is valid.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email is valid, False otherwise
        """
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    def __repr__(self) -> str:
        """String representation of User instance."""
        return f"User(id={self.id}, username='{self.username}', email='{self.email}')"
    
    def __eq__(self, other: object) -> bool:
        """Check equality based on user ID."""
        if not isinstance(other, User):
            return False
        return self.id == other.id if self.id and other.id else False