use api::HttpClient;

#[derive(Clone)]
pub struct Repository {
    pub client: api::HttpClient,
}

impl Repository {
    pub fn new(client: api::HttpClient) -> Self {
        Repository { client }
    }

    pub fn fetch(&self, id: u64) -> Option<api::UserData> {
        self.client.get(id)
    }
}