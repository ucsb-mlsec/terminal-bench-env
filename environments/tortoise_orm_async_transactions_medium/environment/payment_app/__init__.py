from tortoise import transactions
from tortoise.exceptions import DoesNotExist
from payment_app.models import User


async def transfer_funds(from_user_id: int, to_user_id: int, amount: float) -> bool:
    """
    Transfer funds from one user to another atomically.
    
    Args:
        from_user_id: ID of the user to debit from
        to_user_id: ID of the user to credit to
        amount: Amount to transfer
        
    Returns:
        True if transfer completed successfully
        False if transfer cannot complete (insufficient funds, etc.)
        
    Raises:
        DoesNotExist: If either user ID is invalid
        ValueError: If amount is invalid
    """
    if amount <= 0:
        raise ValueError("Transfer amount must be positive")
    
    if from_user_id == to_user_id:
        raise ValueError("Cannot transfer to the same account")
    
    # Use atomic transaction to ensure all-or-nothing behavior
    async with transactions.in_transaction() as conn:
        # Fetch both users within the transaction
        try:
            from_user = await User.get(id=from_user_id).using_db(conn)
        except DoesNotExist:
            raise DoesNotExist(f"User with id {from_user_id} does not exist")
        
        try:
            to_user = await User.get(id=to_user_id).using_db(conn)
        except DoesNotExist:
            raise DoesNotExist(f"User with id {to_user_id} does not exist")
        
        # Check if sender has sufficient funds
        if from_user.balance < amount:
            return False
        
        # Perform the transfer
        from_user.balance -= amount
        to_user.balance += amount
        
        # Save both users within the transaction
        await from_user.save(using_db=conn)
        await to_user.save(using_db=conn)
        
        return True