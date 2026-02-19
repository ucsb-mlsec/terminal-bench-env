#!/usr/bin/env python3

import pandera as pa
from pandera import Column, Check

Stage2Schema = pa.DataFrameSchema(
    {
        # Stage1 columns (maintained with their constraints)
        "transaction_id": Column(
            int,
            checks=[
                Check.greater_than(0),
                Check.unique,
            ],
            nullable=False,
        ),
        "customer_id": Column(
            int,
            checks=[Check.greater_than(0)],
            nullable=False,
        ),
        "transaction_date": Column(
            "datetime64[ns]",
            nullable=False,
        ),
        "amount": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0),
                Check.less_than(10000),
            ],
            nullable=False,
        ),
        "product_category": Column(
            str,
            checks=[
                Check.isin([
                    "Electronics",
                    "Clothing",
                    "Food",
                    "Books",
                    "Home",
                    "Sports",
                ]),
            ],
            nullable=False,
        ),
        "quantity": Column(
            int,
            checks=[
                Check.greater_than_or_equal_to(1),
                Check.less_than_or_equal_to(100),
            ],
            nullable=False,
        ),
        "status": Column(
            str,
            checks=[
                Check.isin(["completed", "pending", "cancelled"]),
            ],
            nullable=False,
        ),
        # Stage2 enrichment columns
        "customer_name": Column(
            str,
            nullable=False,
        ),
        "customer_segment": Column(
            str,
            checks=[
                Check.isin(["Gold", "Silver", "Bronze"]),
            ],
            nullable=False,
        ),
        "region": Column(
            str,
            checks=[
                Check.isin(["North", "South", "East", "West"]),
            ],
            nullable=False,
        ),
        "discount_rate": Column(
            float,
            checks=[
                Check.greater_than_or_equal_to(0.0),
                Check.less_than_or_equal_to(1.0),
            ],
            nullable=False,
        ),
        "revenue": Column(
            float,
            checks=[Check.greater_than(0)],
            nullable=False,
        ),
    },
    coerce=True,
)