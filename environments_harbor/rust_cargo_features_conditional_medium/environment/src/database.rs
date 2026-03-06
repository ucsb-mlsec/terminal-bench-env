[package]
name = "data-processor"
version = "0.1.0"
edition = "2021"

[features]
default = []
database = ["dep:sqlx"]
analytics = []

[dependencies]
sqlx = { version = "0.7", features = ["sqlite", "runtime-tokio-native-tls"], optional = true }
tokio = { version = "1", features = ["full"] }