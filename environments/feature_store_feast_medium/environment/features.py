from datetime import timedelta
from feast import FeatureView, Field, FileSource
from feast.types import Float32, Int64

user_source = FileSource(
    path="data/users.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=["user"],
    schema=[
        Field(name="purchase_count", dtype=Int64),
        Field(name="session_duration", dtype=Float32),
    ],
    source=user_source,
    ttl=timedelta(days=365),
)