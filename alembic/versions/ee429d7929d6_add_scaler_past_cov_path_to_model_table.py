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
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('models', sa.Column('scaler_past_cov_path', sa.Text(), nullable=True, comment='过去协变量缩放器文件在S3中的路径 (key)'))
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('models', 'scaler_past_cov_path')
    # ### end Alembic commands ###
