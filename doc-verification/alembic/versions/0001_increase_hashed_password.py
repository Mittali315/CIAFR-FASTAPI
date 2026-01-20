"""increase hashed_password length

Revision ID: 0001_increase_hashed_password
Revises: 
Create Date: 2025-12-17 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '0001_increase_hashed_password'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column('users', 'hashed_password', existing_type=sa.String(length=65), type_=sa.String(length=200), existing_nullable=False)


def downgrade() -> None:
    op.alter_column('users', 'hashed_password', existing_type=sa.String(length=200), type_=sa.String(length=65), existing_nullable=False)
