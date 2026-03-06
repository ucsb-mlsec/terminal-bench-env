USE recovered_db;

DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    id INT PRIMARY KEY AUTO_INCREMENT,
    customer_name VARCHAR(100)
    email VARCHAR(100) UNIQUE,
    phone VARCHAR(20),
    address TEXT,
    registration_date DATE,
    status VARCHAR(20)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO customers VALUES (1, 'John Smith', 'john.smith@email.com', '555-0101', '123 Main Street, Boston, MA', '2023-01-15', 'active');
INSERT INTO customers VALUES (2, 'Jane Doe', 'jane.doe@email.com', '555-0102', '456 Oak Avenue, New York, NY', '2023-01-20', 'active')
INSERT INTO customers VALUES (3, 'Robert Johnson', 'robert.j@email.com', '555-0103', '789 Pine Road, Los Angeles, CA', '2023-02-01', 'active');
INSERT INTO customers VALUES (4, 'Mary Williams', 'mary.w@email.com', '555-0104', '321 Elm Street, Chicago, IL', '2023-02-10', 'active');
INSERT INTO customers VALUES (5, 'Michael Brown', 'michael.b@email.com', '555-0105', '654 Maple Drive, Houston, TX', '2023-02-15', 'inactive')
INSERT INTO customers VALUES (6, 'Patricia Davis', 'patricia.d@email.com', '555-0106', '987 Cedar Lane, Phoenix, AZ', '2023-03-01', 'active');
INSERT INTO customers VALUES (7, 'James Miller', 'james.m@email.com', '555-0107', '147 Birch Avenue, Philadelphia, PA', '2023-03-05', 'active');
INSERT INTO customers VALUES (8, 'Jennifer Wilson', 'jennifer.w@email.com', '555-0108', '258 Spruce Street, San Antonio, TX', '2023-03-10', 'active')
INSERT INTO customers VALUES (9, 'David Moore', 'david.m@email.com', '555-0109', '369 Walnut Road, San Diego, CA', '2023-03-15', 'active');
INSERT INTO customers VALUES (10, 'Linda Taylor', 'linda.t@email.com', '555-0110', '741 Cherry Boulevard, Dallas, TX', '2023-03-20', 'active');
INSERT INTO customers VALUES (11, 'Richard Anderson', 'richard.a@email.com', '555-0111', '852 Ash Court, San Jose, CA', '2023-04-01', 'active');
INSERT INTO customers VALUES (5, 'Michael Brown', 'michael.b@email.com', '555-0105', '654 Maple Drive, Houston, TX', '2023-02-15', 'inactive');
INSERT INTO customers VALUES (12, 'Barbara Thomas', 'barbara.t@email.com', '555-0112', '963 Willow Way, Austin, TX', '2023-04-05', 'active')
INSERT INTO customers VALUES (13, 'Joseph Jackson', 'joseph.j@email.com', '555-0113', '159 Poplar Place, Jacksonville, FL', '2023-04-10', 'active');
INSERT INTO customers VALUES (14, 'Susan White', 'susan.w@email.com', '555-0114', '357 Hickory Hill, Fort Worth, TX', '2023-04-15', 'inactive');
INSERT INTO customers VALUES (15, 'Thomas Harris', 'thomas.h@email.com', '555-0115', '
INSERT INTO customers VALUES (16, 'Sarah Martin', 'sarah.m@email.com', '555-0116', '753 Sycamore Street, Columbus, OH', '2023-05-01', 'active');
INSERT INTO customers VALUES (17, 'Charles Thompson', 'charles.t@email.com', '555-0117', '951 Magnolia Avenue, Charlotte, NC', '2023-05-05', 'active');
INSERT INTO customers VALUES (18, 'Nancy Garcia', 'nancy.g@email.com', '555-0118', '159 Dogwood Drive, Indianapolis, IN', '2023-05-10', 'active')
INSERT INTO customers VALUES (19, 'Christopher Martinez', 'chris.m@email.com', '555-0119', '357 Redwood Road, San Francisco, CA', '2023-05-15', 'active');
INSERT INTO customers VALUES (20, 'Lisa Rodriguez', 'lisa.r@email.com', '555-0120', '753 Cypress Court, Seattle, WA', '2023-05-20', 'active');
INSERT INTO customers VALUES (21, 'Daniel Robinson', 'daniel.r@email.com', '555-0121', '951 Palm Place, Denver, CO', '2023-06-01', 'inactive')
INSERT INTO customers VALUES (22, 'Karen Clark', 'karen.c@email.com', '555-0122', '147 Fir Lane, Washington, DC', '2023-06-05', 'active');
INSERT INTO customers VALUES (23, 'Matthew Lewis', 'matthew.l@email.com', '555-0123', '258 Beech Boulevard, Boston, MA', '2023-06-10', 'active');
INSERT INTO customers VALUES (24, 'Betty Lee', 'betty.l@email.com', '555-0124', '369 Alder Avenue, Nashville, TN', '2023-06-15', 'active');
INSERT INTO customers VALUES (25, 'Mark Walker', 'mark.w@email.com', '555-0125', '741 Juniper Way, El Paso, TX', '2023-06-20', 'active')
INSERT INTO customers VALUES (26, 'Dorothy Hall', 'dorothy.h@email.com', '555-0126', '852 Sequoia Street, Detroit, MI', '2023-07-01', 'active');
INSERT INTO customers VALUES (27, 'Paul Allen', 'paul.a@email.com', '555-0127', '963 Cottonwood Circle, Memphis, TN', '2023-07-05', 'active');
INSERT INTO customers VALUES (28, 'Sandra Young', 'sandra.y@email.com', '555-0128', 'It's a nice place at 159 O'Reilly Street, Portland, OR', '2023-07-10', 'active');
INSERT INTO customers VALUES (22, 'Karen Clark', 'karen.c@email.com', '555-0122', '147 Fir Lane, Washington, DC', '2023-06-05', 'active');
INSERT INTO customers VALUES (29, 'Steven Hernandez', 'steven.h@email.com', '555-0129', '357 Basswood Avenue, Las Vegas, NV', '2023-07-15', 'inactive');
INSERT INTO customers VALUES (30, 'Ashley King', 'ashley.k@email.com', '555-0130', '753 Chestnut Road, Louisville, KY', '2023-07-20', 'active')
INSERT INTO customers VALUES (31, 'Andrew Wright', 'andrew.w@email.com', '555-0131', '951 Butternut Drive, Baltimore, MD', '2023-08-01', 'active');
INSERT INTO customers VALUES (32, 'Kimberly Lopez', 'kimberly.l@email.com', '555-0132', '147 Laurel Lane, Milwaukee, WI', '2023-08-05', 'active');
INSERT INTO customers VALUES (33, 'Joshua Hill', 'joshua.h@email.com', '555-0133', '258 Hawthorn Place, Albuquerque, NM', '2023-08-10', 'active');
INSERT INTO customers VALUES (34, 'Donna Scott', 'donna.s@email.com', '555-0134', '369 Ironwood Court, Tucson, AZ', '2023-08-15', 'inactive')
INSERT INTO customers VALUES (35, 'Kenneth Green', 'kenneth.g@email.com', '555-0135', '741 Mesquite Way, Fresno, CA', '2023-08-20', 'active');
INSERT INTO customers VALUES (36, 'Carol Adams', 'carol.a@email.com', '555-0136', '852 Acacia Avenue, Sacramento, CA', '2023-09-01', 'active');
INSERT INTO customers VALUES (37, 'Kevin Baker', 'kevin.b@email.com', '555-0137', '963 Hemlock Street, Long Beach, CA', '2023-09-05', 'active')
INSERT INTO customers VALUES (38, 'Michelle Gonzalez', 'michelle.g@email.com', '555-0138', '159 Linden Boulevard, Kansas City, MO', '2023-09-10', 'active');
INSERT INTO customers VALUES (39, 'Brian Nelson', 'brian.n@email.com', '555-0139', '357 Buckeye Road, Mesa, AZ', '2023-09-15', 'active');
INSERT INTO customers VALUES (40, 'Emily Carter', 'emily.c@email.com', '555-0140', '753 Sumac Drive, Atlanta, GA', '2023-09-20', 'inactive');
INSERT INTO customers VALUES (41, 'George Mitchell', 'george.m@email.com', '555-0141', '951 Persimmon Place, Virginia Beach, VA', '2023-10-01', 'active')
INSERT INTO customers VALUES (42, 'Amanda Perez', 'amanda.p@email.com', '555-0142', '147 Mulberry Court, Omaha, NE', '2023-10-05', 'active');
INSERT INTO customers VALUES (43, 'Ronald Roberts', 'ronald.r@email.com', '555-0143', '258 Catalpa Lane, Oakland, CA', '2023-10-10', 'active');
INSERT INTO customers VALUES (44, 'Michelle Turner', 'michelle.t@email.com', '555-0144', '369 Plum Way, Miami, FL', '2023-10-15', 'active');
INSERT INTO customers VALUES (45, 'Edward Phillips', 'edward.p@email.com', '555-0145', '741 Peach Street, Tulsa, OK', '2023-10-20', 'active')
INSERT INTO customers VALUES (46, 'Deborah Campbell', 'deborah.c@email.com', '555-0146', '852 Pear Avenue, Minneapolis, MN', '2023-11-01', 'active');
INSERT INTO customers VALUES (47, 'Jason Parker', 'jason.p@email.com', '555-0147', '963 Apple Boulevard, Cleveland, OH', '2023-11-05', 'inactive');
INSERT INTO customers VALUES (48, 'Stephanie Evans', 'stephanie.e@email.com', '555-0148', '159 Apricot Road, Wichita, KS', '2023-11-10', 'active')
INSERT INTO customers VALUES (49, 'Ryan Edwards', 'ryan.e@email.com', '555-0149', '357 Banana Drive, New Orleans, LA', '2023-11-15', 'active');
INSERT INTO customers VALUES (50, 'Rebecca Collins', 'rebecca.c@email.com', '555-0150', '753