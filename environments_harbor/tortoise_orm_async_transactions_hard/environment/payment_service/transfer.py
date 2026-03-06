import sqlite3

def transfer(conn, from_account_id, to_account_id, amount):
    """
    Transfer money from one account to another.
    
    Args:
        conn: Database connection
        from_account_id: ID of account to transfer from
        to_account_id: ID of account to transfer to
        amount: Amount to transfer
    """
    cursor = conn.cursor()
    
    # Fetch current balance of from_account
    cursor.execute("SELECT balance FROM accounts WHERE id = ?", (from_account_id,))
    from_balance = cursor.fetchone()[0]
    
    # Fetch current balance of to_account
    cursor.execute("SELECT balance FROM accounts WHERE id = ?", (to_account_id,))
    to_balance = cursor.fetchone()[0]
    
    # Calculate new balances
    new_from_balance = from_balance - amount
    new_to_balance = to_balance + amount
    
    # Update from_account
    cursor.execute("UPDATE accounts SET balance = ? WHERE id = ?", (new_from_balance, from_account_id))
    
    # Update to_account
    cursor.execute("UPDATE accounts SET balance = ? WHERE id = ?", (new_to_balance, to_account_id))
    
    # Commit the transaction
    conn.commit()
    
    cursor.close()