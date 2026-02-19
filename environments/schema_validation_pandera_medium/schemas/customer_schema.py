#!/usr/bin/env python3

import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Customer transaction schema with INTENTIONAL BUGS for testing purposes

customer_schema = DataFrameSchema(
    {
        # Bug 1: Using <= 0 instead of > 0 for positive validation
        "customer_id": Column(
            pa.Int,
            checks=Check(lambda x: x <= 0),  # Should be > 0
            nullable=False
        ),
        
        # Bug 2: Reversed min/max values (min=5000, max=10)
        # Bug 3: Using pa.Int instead of pa.Float for decimal amounts
        "transaction_amount": Column(
            pa.Int,  # Should be pa.Float
            checks=Check.in_range(min_value=5000.00, max_value=10.00),  # Values are reversed
            nullable=False
        ),
        
        # Bug 4: Wrong datetime format string and missing coerce parameter
        # Should be "%Y-%m-%d" and coerce=True
        "transaction_date": Column(
            pa.DateTime,
            checks=Check(lambda x: True),  # Placeholder check
            nullable=False,
            # Missing coerce=True parameter and wrong format
        ),
        
        # Bug 5: Missing Check constraint for age bounds
        # Should validate ages between 18-85
        "customer_age": Column(
            pa.Int,
            nullable=False
            # Missing: checks=Check.in_range(min_value=18, max_value=85)
        ),
        
        # Bug 6: Incorrect regex pattern for email validation
        # Current pattern is malformed and won't match valid emails
        "email": Column(
            pa.String,
            checks=Check.str_matches(r"^[a-z]$"),  # Should be proper email regex like r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
            nullable=False
        )
    },
    strict=True
)