#!/usr/bin/env python3

import os
import subprocess
import tempfile

# Create the Alembic directory structure
os.makedirs('alembic/versions', exist_ok=True)

# Create the initial migration file
with open('alembic/versions/a1b2c3d4e5f6_add_users_table.py', 'w') as f:
    f.write('''"""add_users_table

Revision ID: a1b2c3d4e5f6
Revises: 
Create Date: 2024-01-15 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('username', sa.String(length=50), nullable=False),
        sa.Column('email', sa.String(length=100), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('users')
''')

# Create the first branching migration (add email index)
with open('alembic/versions/1a2b3c4d5e6f_add_user_email_index.py', 'w') as f:
    f.write('''"""add_user_email_index

Revision ID: 1a2b3c4d5e6f
Revises: a1b2c3d4e5f6
Create Date: 2024-01-16 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a2b3c4d5e6f'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_index('ix_users_email', 'users', ['email'])


def downgrade():
    op.drop_index('ix_users_email', table_name='users')
''')

# Create the second branching migration (add status column)
with open('alembic/versions/9z8y7x6w5v4u_add_user_status_column.py', 'w') as f:
    f.write('''"""add_user_status_column

Revision ID: 9z8y7x6w5v4u
Revises: a1b2c3d4e5f6
Create Date: 2024-01-16 11:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '9z8y7x6w5v4u'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('users', sa.Column('status', sa.String(length=20), nullable=True))


def downgrade():
    op.drop_column('users', 'status')
''')

# Create alembic.ini if it doesn't exist
if not os.path.exists('alembic.ini'):
    with open('alembic.ini', 'w') as f:
        f.write('''[alembic]
script_location = alembic
sqlalchemy.url = sqlite:///test.db

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console
qualname =

[logger_sqlalchemy]
level = WARN
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
''')

# Create alembic/env.py if it doesn't exist
if not os.path.exists('alembic/env.py'):
    with open('alembic/env.py', 'w') as f:
        f.write('''from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

def run_migrations_online():
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()

run_migrations_online()
''')

# Create the merge migration using alembic merge command
result = subprocess.run(
    ['alembic', 'merge', '-m', 'merge_branches', '1a2b3c4d5e6f', '9z8y7x6w5v4u'],
    capture_output=True,
    text=True
)

# Extract the merge revision ID from the output
merge_revision = None
for line in result.stdout.split('\n'):
    if 'Generating' in line and 'alembic/versions/' in line:
        # Extract revision ID from path like: alembic/versions/abc123_merge_branches.py
        parts = line.split('alembic/versions/')[1].split('_')[0]
        merge_revision = parts
        break

# If we couldn't get it from the command output, read the latest merge file
if not merge_revision:
    import glob
    merge_files = glob.glob('alembic/versions/*_merge_branches.py')
    if merge_files:
        merge_file = merge_files[0]
        with open(merge_file, 'r') as f:
            for line in f:
                if line.startswith("revision = "):
                    merge_revision = line.split("'")[1]
                    break

# Write the solution to /tmp/solution.txt
with open('/tmp/solution.txt', 'w') as f:
    f.write(f'merge_revision={merge_revision}\n')
    f.write('head_count=1\n')

print(f"Solution written to /tmp/solution.txt")
print(f"Merge revision: {merge_revision}")