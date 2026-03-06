"""add_comments_table

Revision ID: c3d4e5f6g7h8
Revises: a1b2c3d4e5f6
Create Date: 2024-01-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c3d4e5f6g7h8'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table('comments',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('post_id', sa.Integer(), sa.ForeignKey('posts.id'))
    )


def downgrade():
    op.drop_table('comments')