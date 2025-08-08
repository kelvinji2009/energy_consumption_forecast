"""Add status and S3 paths to Model table

Revision ID: 2fb253a77cdb
Revises: 79108569c0f0
Create Date: 2025-07-07 03:30:16.936503

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2fb253a77cdb'
down_revision: Union[str, None] = '79108569c0f0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This migration is now redundant because the initial migration
    # creates the table with all necessary columns.
    pass


def downgrade() -> None:
    # This migration is now redundant.
    pass