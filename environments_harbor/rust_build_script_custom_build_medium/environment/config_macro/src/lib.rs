include!(concat!(env!("OUT_DIR"), "/generated_config.rs"));

pub fn get_app_name() -> &'static str {
    GENERATED_CONFIG.app_name
}

pub fn get_version() -> &'static str {
    GENERATED_CONFIG.version
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_name() {
        assert_eq!(get_app_name(), "MyApplication");
    }

    #[test]
    fn test_version() {
        assert_eq!(get_version(), "1.0.0");
    }
}