use openssl::ssl::SslVersion;
use openssl::hash::MessageDigest;

fn main() {
    // Access SSL version to ensure we link against OpenSSL
    println!("TLS 1.2 version: {:?}", SslVersion::TLS1_2);
    
    // Use MessageDigest to force linking to OpenSSL crypto functions
    let md = MessageDigest::sha256();
    println!("Using digest: {:?}", md);
    
    println!("OpenSSL linked successfully!");
}