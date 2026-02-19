pub struct Compressor;

impl Compressor {
    pub fn new() -> Self {
        Compressor
    }

    pub fn compress(&self, data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    pub fn decompress(&self, data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }
}

pub fn init_compressor() {}