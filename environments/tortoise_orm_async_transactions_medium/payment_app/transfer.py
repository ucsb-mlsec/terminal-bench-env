from tortoise.transactions import atomic
from payment_app.models import User


async def transfer_funds(from_user_id: int, to_user_id: int, amount: float) -> bool:
    """
    Transfer funds from one user to another atomically.
    
    Args:
        from_user_id: ID of the user sending money
        to_user_id: ID of the user receiving money
        amount: Amount to transfer
        
    Returns:
        True if transfer successful, False if insufficient funds
        
    Raises:
        ValueError: If amount is invalid or users are the same
        Exception: If user IDs are invalid or users don't exist
    """
    # Validate amount
    if amount <= 0:
        raise ValueError("Transfer amount must be positive")
    
    # Validate users are different
    if from_user_id == to_user_id:
        raise ValueError("Cannot transfer to the same account")
    
    # Use atomic transaction to ensure all-or-nothing behavior
    async with atomic():
        # Fetch both users within the transaction
        from_user = await User.get_or_none(id=from_user_id)
        to_user = await User.get_or_none(id=to_user_id)
        
        # Validate users exist
        if from_user is None:
            raise Exception(f"User with id {from_user_id} not found")
        if to_user is None:
            raise Exception(f"User with id {to_user_id} not found")
        
        # Check sufficient funds
        if from_user.balance < amount:
            return False
        
        # Perform the transfer
        from_user.balance -= amount
        to_user.balance += amount
        
        # Save both accounts within the transaction
        await from_user.save()
        await to_user.save()
    
    return True