"""add_model_training_parameters_table

Revision ID: 13f3f88a257f
Revises: 1ecb68f46cbf
Create Date: 2025-08-22 03:36:33.577856

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '13f3f88a257f'
down_revision: Union[str, None] = '1ecb68f46cbf'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 创建训练参数表
    op.create_table('model_training_parameters',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('parameter_name', sa.String(length=100), nullable=False),
        sa.Column('parameter_value', sa.Text(), nullable=False),
        sa.Column('parameter_type', sa.String(length=50), nullable=False),
        sa.Column('parameter_category', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['models.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('model_id', 'parameter_name', name='uq_model_param'),
        comment='模型训练参数存储表'
    )
    
    # 创建索引
    op.create_index('ix_training_params_model_id', 'model_training_parameters', ['model_id'])
    op.create_index('ix_training_params_name', 'model_training_parameters', ['parameter_name'])
    op.create_index('ix_training_params_category', 'model_training_parameters', ['parameter_category'])


def downgrade() -> None:
    # 删除索引
    op.drop_index('ix_training_params_category', table_name='model_training_parameters')
    op.drop_index('ix_training_params_name', table_name='model_training_parameters')
    op.drop_index('ix_training_params_model_id', table_name='model_training_parameters')
    
    # 删除表
    op.drop_table('model_training_parameters')
