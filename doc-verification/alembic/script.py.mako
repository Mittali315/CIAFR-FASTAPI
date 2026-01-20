"""
Revision script template for Alembic
"""

from alembic import op
import sqlalchemy as sa

{{ imports }}

revision = '{{ up_revision }}'
down_revision = {{ down_revision | repr }}
branch_labels = {{ branch_labels | repr }}
depends_on = {{ depends_on | repr }}


def upgrade():
    {{ upgrade_ops }}


def downgrade():
    {{ downgrade_ops }}
