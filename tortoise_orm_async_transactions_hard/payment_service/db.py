import sqlite3

def get_connection():
    return sqlite3.connect(':memory:', check_same_thread=False)

def init_db(conn):
    conn.execute('''
        CREATE TABLE IF NOT EXISTS accounts (
            id INTEGER PRIMARY KEY,
            balance REAL
        )
    ''')
    conn.commit()

def reset_accounts(conn):
    conn.execute('DELETE FROM accounts')
    conn.execute('INSERT INTO accounts (id, balance) VALUES (1, 1000.0)')
    conn.execute('INSERT INTO accounts (id, balance) VALUES (2, 1000.0)')
    conn.commit()