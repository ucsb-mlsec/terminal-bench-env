DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS books;

CREATE TABLE books (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    title VARCHAR(255) NOT NULL,
    author VARCHAR(255) NOT NULL
);

CREATE TABLE reviews (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    review_text VARCHAR(1000),
    rating INTEGER NOT NULL,
    book_id BIGINT NOT NULL,
    FOREIGN KEY (book_id) REFERENCES books(id)
);

INSERT INTO books (title, author) VALUES ('To Kill a Mockingbird', 'Harper Lee');
INSERT INTO books (title, author) VALUES ('1984', 'George Orwell');
INSERT INTO books (title, author) VALUES ('The Great Gatsby', 'F. Scott Fitzgerald');
INSERT INTO books (title, author) VALUES ('Pride and Prejudice', 'Jane Austen');

INSERT INTO reviews (review_text, rating, book_id) VALUES ('A timeless classic that explores themes of racial injustice and moral growth. Absolutely brilliant.', 5, 1);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('One of the most important books in American literature. Scout is an unforgettable narrator.', 5, 1);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('Powerful and moving. A must-read for everyone.', 4, 1);

INSERT INTO reviews (review_text, rating, book_id) VALUES ('A chilling dystopian masterpiece that feels more relevant today than ever before.', 5, 2);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('Orwell predicted the future with frightening accuracy. Big Brother is watching!', 5, 2);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('Dark and thought-provoking. Made me question everything about surveillance and freedom.', 4, 2);

INSERT INTO reviews (review_text, rating, book_id) VALUES ('Beautiful prose and a tragic tale of the American Dream. Fitzgerald at his finest.', 5, 3);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('The parties, the glamour, the heartbreak - a perfect snapshot of the Jazz Age.', 4, 3);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('Short but profound. Every sentence is carefully crafted.', 5, 3);

INSERT INTO reviews (review_text, rating, book_id) VALUES ('Witty, romantic, and surprisingly modern. Elizabeth Bennet is one of literature greatest heroines.', 5, 4);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('A delightful comedy of manners with sharp social commentary. Jane Austen genius shines through.', 5, 4);
INSERT INTO reviews (review_text, rating, book_id) VALUES ('Charming and entertaining, though the pacing is a bit slow at times.', 4, 4);