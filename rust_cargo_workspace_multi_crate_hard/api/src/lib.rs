use core::{User, UserService};

#[derive(Debug, Clone)]
pub struct UserData {
    pub id: u64,
    pub name: String,
}

pub struct HttpClient {
    service: Box<dyn UserService>,
}

impl Clone for HttpClient {
    fn clone(&self) -> Self {
        HttpClient {
            service: self.service.clone_box(),
        }
    }
}

impl HttpClient {
    pub fn get(&self, id: u64) -> Option<UserData> {
        self.service.get_user(id).map(|user| UserData {
            id: user.id,
            name: user.name,
        })
    }
}

pub fn create_client(service: Box<dyn UserService>) -> HttpClient {
    HttpClient { service }
}