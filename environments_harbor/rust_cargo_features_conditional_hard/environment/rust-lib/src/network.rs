use crate::logger;

pub struct NetworkClient {
    host: String,
}

impl NetworkClient {
    pub fn new() -> Self {
        logger::init_logger();
        NetworkClient {
            host: String::from("localhost"),
        }
    }

    pub fn connect(&self) {
        logger::log(&format!("Connecting to {}", self.host));
    }
}

pub fn start_network() {
    logger::log("Starting network subsystem");
    let client = NetworkClient::new();
    client.connect();
}

#[cfg(feature = "compresion")]
fn helper_function() {
    // This shouldn't be here
}