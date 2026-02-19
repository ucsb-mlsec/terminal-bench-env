use storage::Repository;

#[derive(Debug, Clone)]
pub struct User {
    pub id: u64,
    pub name: String,
}

pub trait UserService {
    fn get_user(&self, id: u64) -> Option<User>;
}

pub struct UserServiceImpl {
    pub repo: Repository,
}

impl UserService for UserServiceImpl {
    fn get_user(&self, id: u64) -> Option<User> {
        self.repo.fetch(id)
    }
}