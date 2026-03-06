#!/usr/bin/env python3

import pandera as pa
from pandera import Column, Check, DataFrameSchema

Stage3Schema = DataFrameSchema(
    columns={
        "customer_id": Column(
            int,
            checks=[
                Check.greater_than(0),
            ],
            unique=True,
            nullable=False
        ),
        "product_category": Column(
            str,
            checks=[
                Check.isin(['Electronics', 'Clothing', 'Food', 'Books', 'Home'])
            ],
            nullable=False
        ),
        "total_amount": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0)
            ],
            nullable=False
        ),
        "transaction_count": Column(
            int,
            checks=[
                Check.greater_than(0)
            ],
            nullable=False
        ),
        "avg_quantity": Column(
            float,
            checks=[
                Check.greater_than(0)
            ],
            nullable=False
        ),
        "max_amount": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0)
            ],
            nullable=False
        ),
        "min_amount": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0)
            ],
            nullable=False
        ),
    },
    checks=[
        Check(lambda df: (df['max_amount'] >= df['min_amount']).all(), 
              error="max_amount must be >= min_amount")
    ],
    coerce=True,
    strict=False
)