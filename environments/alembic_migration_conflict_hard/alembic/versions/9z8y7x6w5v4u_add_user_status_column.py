"""add_user_status_column

Revision ID: 9z8y7x6w5v4u
Revises: a1b2c3d4e5f6
Create Date: 2024-02-02 09:15:00.000000

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