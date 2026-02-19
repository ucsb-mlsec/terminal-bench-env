pub struct DataPoint {
    pub id: u32,
    pub value: f64,
}

impl DataPoint {
    pub fn new(id: u32, value: f64) -> Self {
        DataPoint { id, value }
    }
}

pub fn process_basic_data() {
    println!("Processing basic data");
}