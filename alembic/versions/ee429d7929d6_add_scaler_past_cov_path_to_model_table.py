"""Add scaler_past_cov_path to Model table

Revision ID: ee429d7929d6
Revises: 2fb253a77cdb
Create Date: 2025-07-07 06:23:18.083397

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ee429d7929d6'
down_revision: Union[str, None] = '2fb253a77cdb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This migration is now redundant because the initial migration
    # creates the table with all necessary columns.
    pass


def downgrade() -> None:
    # This migration is now redundant.
    pass