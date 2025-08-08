"""Add detector_path to models table

Revision ID: 3de837a80144
Revises: ee429d7929d6
Create Date: 2025-07-07 12:13:37.431470

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3de837a80144'
down_revision: Union[str, None] = 'ee429d7929d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # This migration is now redundant because the initial migration
    # creates the table with all necessary columns.
    pass


def downgrade() -> None:
    # This migration is now redundant.
    pass