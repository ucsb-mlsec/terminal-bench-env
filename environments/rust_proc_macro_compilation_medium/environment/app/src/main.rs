use derive_macro::MyTrait;

trait MyTrait {
    fn describe(&self) -> String;
}

#[derive(MyTrait)]
struct Person {
    name: String,
    age: u32,
}

#[derive(MyTrait)]
struct Book {
    title: String,
    author: String,
    pages: usize,
}

#[derive(MyTrait)]
struct Car {
    brand: String,
    model: String,
    year: u32,
}

fn main() {
    let person = Person {
        name: String::from("Alice"),
        age: 30,
    };

    let book = Book {
        title: String::from("The Rust Programming Language"),
        author: String::from("Steve Klabnik and Carol Nichols"),
        pages: 552,
    };

    let car = Car {
        brand: String::from("Toyota"),
        model: String::from("Camry"),
        year: 2023,
    };

    println!("{}", person.describe());
    println!("{}", book.describe());
    println!("{}", car.describe());
}