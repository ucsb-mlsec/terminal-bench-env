#!/usr/bin/env python3
"""Stage 1 Schema Definition - Transaction Ingestion"""

import pandera as pa
from pandera.typing import Series, DataFrame
from typing import Optional
import pandas as pd

class Stage1Schema(pa.DataFrameModel):
    """Schema for stage 1 transaction ingestion data"""
    
    transaction_id: Series[int] = pa.Field(
        gt=0,
        unique=True,
        nullable=False,
        coerce=True,
        description="Unique transaction identifier"
    )
    
    customer_id: Series[int] = pa.Field(
        gt=0,
        nullable=False,
        coerce=True,
        description="Customer identifier"
    )
    
    transaction_date: Series[pd.Timestamp] = pa.Field(
        nullable=False,
        coerce=True,
        description="Transaction timestamp"
    )
    
    amount: Series[float] = pa.Field(
        ge=0.0,
        lt=10000.0,
        nullable=False,
        coerce=True,
        description="Transaction amount"
    )
    
    product_category: Series[str] = pa.Field(
        isin=['Electronics', 'Clothing', 'Food', 'Books', 'Home'],
        nullable=False,
        coerce=True,
        description="Product category"
    )
    
    quantity: Series[int] = pa.Field(
        ge=1,
        le=100,
        nullable=False,
        coerce=True,
        description="Quantity of items"
    )
    
    status: Series[str] = pa.Field(
        isin=['completed', 'pending', 'cancelled'],
        nullable=False,
        coerce=True,
        description="Transaction status"
    )
    
    class Config:
        """Pandera config"""
        coerce = True
        strict = False
        ordered = False
