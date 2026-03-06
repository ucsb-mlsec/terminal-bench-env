-- Sample Customer Engagement Data for Census Reverse ETL Testing
-- This script creates a table with various data quality issues to test the pipeline

-- Create schema
CREATE SCHEMA IF NOT EXISTS analytics;

-- Drop table if exists for clean setup
DROP TABLE IF EXISTS analytics.customer_engagement;

-- Create customer engagement table
CREATE TABLE analytics.customer_engagement (
    customer_id INTEGER PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(50),
    engagement_score INTEGER,
    last_activity_date DATE,
    status VARCHAR(50),
    created_at TIMESTAMP,
    updated_at TIMESTAMP
);

-- Insert sample data with various scenarios

-- VALID RECORDS: High engagement, active, recent activity (60% = ~72 records)
INSERT INTO analytics.customer_engagement VALUES
(1, 'sarah.johnson@email.com', 'Sarah', 'Johnson', '+14155552671', 85, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '90 days', NOW()),
(2, 'michael.chen@company.com', 'Michael', 'Chen', '+14155552672', 92, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '120 days', NOW()),
(3, 'jessica.williams@enterprise.com', 'Jessica', 'Williams', '+14155552673', 78, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '60 days', NOW()),
(4, 'david.martinez@business.com', 'David', 'Martinez', '+14155552674', 88, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '45 days', NOW()),
(5, 'emily.brown@corp.com', 'Emily', 'Brown', '+14155552675', 95, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '30 days', NOW()),
(6, 'james.davis@startup.com', 'James', 'Davis', '+14155552676', 71, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '75 days', NOW()),
(7, 'amanda.taylor@tech.com', 'Amanda', 'Taylor', '+14155552677', 83, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '50 days', NOW()),
(8, 'robert.anderson@solutions.com', 'Robert', 'Anderson', '+14155552678', 67, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '40 days', NOW()),
(9, 'jennifer.thomas@digital.com', 'Jennifer', 'Thomas', '+14155552679', 90, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '55 days', NOW()),
(10, 'william.jackson@ventures.com', 'William', 'Jackson', '+14155552680', 76, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '65 days', NOW()),
(11, 'lisa.white@global.com', 'Lisa', 'White', '+14155552681', 82, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '70 days', NOW()),
(12, 'daniel.harris@systems.com', 'Daniel', 'Harris', '+14155552682', 89, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '35 days', NOW()),
(13, 'michelle.martin@cloud.com', 'Michelle', 'Martin', '+14155552683', 74, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '80 days', NOW()),
(14, 'christopher.garcia@partners.com', 'Christopher', 'Garcia', '+14155552684', 86, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '25 days', NOW()),
(15, 'karen.rodriguez@analytics.com', 'Karen', 'Rodriguez', '+14155552685', 79, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '95 days', NOW()),
(16, 'matthew.wilson@media.com', 'Matthew', 'Wilson', '+14155552686', 91, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '100 days', NOW()),
(17, 'nancy.moore@services.com', 'Nancy', 'Moore', '+14155552687', 68, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '45 days', NOW()),
(18, 'joseph.lee@consulting.com', 'Joseph', 'Lee', '+14155552688', 84, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '60 days', NOW()),
(19, 'susan.walker@agency.com', 'Susan', 'Walker', '+14155552689', 77, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '50 days', NOW()),
(20, 'thomas.hall@innovations.com', 'Thomas', 'Hall', '+14155552690', 93, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '75 days', NOW()),
(21, 'betty.allen@platforms.com', 'Betty', 'Allen', '+14155552691', 72, CURRENT_DATE - 16, 'active', NOW() - INTERVAL '40 days', NOW()),
(22, 'charles.young@networks.com', 'Charles', 'Young', '+14155552692', 87, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '85 days', NOW()),
(23, 'donna.king@software.com', 'Donna', 'King', '+14155552693', 80, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '55 days', NOW()),
(24, 'steven.wright@apps.com', 'Steven', 'Wright', '+14155552694', 75, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '65 days', NOW()),
(25, 'carol.lopez@design.com', 'Carol', 'Lopez', '+14155552695', 94, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '70 days', NOW()),
(26, 'paul.hill@creative.com', 'Paul', 'Hill', '+14155552696', 69, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '30 days', NOW()),
(27, 'maria.scott@dev.com', 'Maria', 'Scott', '+14155552697', 81, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '90 days', NOW()),
(28, 'mark.green@web.com', 'Mark', 'Green', '+14155552698', 88, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '35 days', NOW()),
(29, 'sandra.adams@mobile.com', 'Sandra', 'Adams', '+14155552699', 73, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '80 days', NOW()),
(30, 'kevin.baker@ecommerce.com', 'Kevin', 'Baker', '+14155552700', 85, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '25 days', NOW()),
(31, 'laura.gonzalez@retail.com', 'Laura', 'Gonzalez', '+14155552701', 90, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '95 days', NOW()),
(32, 'ronald.nelson@finance.com', 'Ronald', 'Nelson', '+14155552702', 66, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '100 days', NOW()),
(33, 'sharon.carter@banking.com', 'Sharon', 'Carter', '+14155552703', 82, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '45 days', NOW()),
(34, 'jason.mitchell@invest.com', 'Jason', 'Mitchell', '+14155552704', 78, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '60 days', NOW()),
(35, 'deborah.perez@wealth.com', 'Deborah', 'Perez', '+14155552705', 91, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '50 days', NOW()),
(36, 'brian.roberts@insurance.com', 'Brian', 'Roberts', '+14155552706', 74, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '75 days', NOW()),
(37, 'linda.turner@healthcare.com', 'Linda', 'Turner', '+14155552707', 86, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '40 days', NOW()),
(38, 'george.phillips@medical.com', 'George', 'Phillips', '+14155552708', 70, CURRENT_DATE - 16, 'active', NOW() - INTERVAL '85 days', NOW()),
(39, 'helen.campbell@pharma.com', 'Helen', 'Campbell', '+14155552709', 83, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '55 days', NOW()),
(40, 'eric.parker@biotech.com', 'Eric', 'Parker', '+14155552710', 89, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '65 days', NOW()),
(41, 'cynthia.evans@research.com', 'Cynthia', 'Evans', '+14155552711', 76, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '70 days', NOW()),
(42, 'timothy.edwards@labs.com', 'Timothy', 'Edwards', '+14155552712', 92, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '30 days', NOW()),
(43, 'angela.collins@science.com', 'Angela', 'Collins', '+14155552713', 68, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '90 days', NOW()),
(44, 'jeffrey.stewart@education.com', 'Jeffrey', 'Stewart', '+14155552714', 84, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '35 days', NOW()),
(45, 'brenda.sanchez@learning.com', 'Brenda', 'Sanchez', '+14155552715', 80, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '80 days', NOW()),
(46, 'scott.morris@training.com', 'Scott', 'Morris', '+14155552716', 87, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '25 days', NOW()),
(47, 'kathleen.rogers@courses.com', 'Kathleen', 'Rogers', '+14155552717', 72, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '95 days', NOW()),
(48, 'raymond.reed@university.com', 'Raymond', 'Reed', '+14155552718', 93, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '100 days', NOW()),
(49, 'rebecca.cook@academy.com', 'Rebecca', 'Cook', '+14155552719', 77, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '45 days', NOW()),
(50, 'gregory.morgan@institute.com', 'Gregory', 'Morgan', '+14155552720', 85, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '60 days', NOW()),
(51, 'carolyn.bell@travel.com', 'Carolyn', 'Bell', '+14155552721', 81, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '50 days', NOW()),
(52, 'larry.murphy@hospitality.com', 'Larry', 'Murphy', '+14155552722', 88, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '75 days', NOW()),
(53, 'janet.bailey@hotels.com', 'Janet', 'Bailey', '+14155552723', 73, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '40 days', NOW()),
(54, 'frank.rivera@tourism.com', 'Frank', 'Rivera', '+14155552724', 90, CURRENT_DATE - 16, 'active', NOW() - INTERVAL '85 days', NOW()),
(55, 'barbara.cooper@adventures.com', 'Barbara', 'Cooper', '+14155552725', 67, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '55 days', NOW()),
(56, 'donald.richardson@tours.com', 'Donald', 'Richardson', '+14155552726', 82, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '65 days', NOW()),
(57, 'ruth.cox@events.com', 'Ruth', 'Cox', '+14155552727', 78, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '70 days', NOW()),
(58, 'dennis.howard@catering.com', 'Dennis', 'Howard', '+14155552728', 91, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '30 days', NOW()),
(59, 'judy.ward@venues.com', 'Judy', 'Ward', '+14155552729', 74, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '90 days', NOW()),
(60, 'walter.torres@planning.com', 'Walter', 'Torres', '+14155552730', 86, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '35 days', NOW()),
(61, 'joyce.peterson@production.com', 'Joyce', 'Peterson', '+14155552731', 79, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '80 days', NOW()),
(62, 'patrick.gray@manufacturing.com', 'Patrick', 'Gray', '+14155552732', 89, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '25 days', NOW()),
(63, 'evelyn.ramirez@factory.com', 'Evelyn', 'Ramirez', '+14155552733', 71, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '95 days', NOW()),
(64, 'peter.james@industrial.com', 'Peter', 'James', '+14155552734', 92, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '100 days', NOW()),
(65, 'alice.watson@supply.com', 'Alice', 'Watson', '+14155552735', 76, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '45 days', NOW()),
(66, 'harold.brooks@logistics.com', 'Harold', 'Brooks', '+14155552736', 84, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '60 days', NOW()),
(67, 'frances.kelly@shipping.com', 'Frances', 'Kelly', '+14155552737', 80, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '50 days', NOW()),
(68, 'arthur.sanders@transport.com', 'Arthur', 'Sanders', '+14155552738', 87, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '75 days', NOW()),
(69, 'teresa.price@delivery.com', 'Teresa', 'Price', '+14155552739', 72, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '40 days', NOW()),
(70, 'albert.bennett@freight.com', 'Albert', 'Bennett', '+14155552740', 93, CURRENT_DATE - 16, 'active', NOW() - INTERVAL '85 days', NOW()),
(71, 'doris.wood@cargo.com', 'Doris', 'Wood', '+14155552741', 68, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '55 days', NOW()),
(72, 'joe.barnes@warehouse.com', 'Joe', 'Barnes', '+14155552742', 83, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '65 days', NOW());

-- LOW ENGAGEMENT SCORES: < 50 (20% = ~24 records)
INSERT INTO analytics.customer_engagement VALUES
(73, 'anna.ross@email.com', 'Anna', 'Ross', '+14155552743', 45, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '60 days', NOW()),
(74, 'carl.henderson@company.com', 'Carl', 'Henderson', '+14155552744', 32, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '70 days', NOW()),
(75, 'marie.coleman@enterprise.com', 'Marie', 'Coleman', '+14155552745', 28, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '80 days', NOW()),
(76, 'wayne.jenkins@business.com', 'Wayne', 'Jenkins', '+14155552746', 41, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '90 days', NOW()),
(77, 'jean.perry@corp.com', 'Jean', 'Perry', '+14155552747', 15, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '100 days', NOW()),
(78, 'ralph.powell@startup.com', 'Ralph', 'Powell', '+14155552748', 38, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '50 days', NOW()),
(79, 'gloria.long@tech.com', 'Gloria', 'Long', '+14155552749', 22, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '65 days', NOW()),
(80, 'roy.patterson@solutions.com', 'Roy', 'Patterson', '+14155552750', 49, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '75 days', NOW()),
(81, 'joan.hughes@digital.com', 'Joan', 'Hughes', '+14155552751', 35, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '85 days', NOW()),
(82, 'jack.flores@ventures.com', 'Jack', 'Flores', '+14155552752', 19, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '95 days', NOW()),
(83, 'phyllis.washington@global.com', 'Phyllis', 'Washington', '+14155552753', 43, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '45 days', NOW()),
(84, 'eugene.butler@systems.com', 'Eugene', 'Butler', '+14155552754', 27, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '55 days', NOW()),
(85, 'rose.simmons@cloud.com', 'Rose', 'Simmons', '+14155552755', 36, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '60 days', NOW()),
(86, 'russell.foster@partners.com', 'Russell', 'Foster', '+14155552756', 44, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '70 days', NOW()),
(87, 'ann.gonzales@analytics.com', 'Ann', 'Gonzales', '+14155552757', 21, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '80 days', NOW()),
(88, 'louis.bryant@media.com', 'Louis', 'Bryant', '+14155552758', 39, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '90 days', NOW()),
(89, 'katherine.alexander@services.com', 'Katherine', 'Alexander', '+14155552759', 16, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '100 days', NOW()),
(90, 'jerry.russell@consulting.com', 'Jerry', 'Russell', '+14155552760', 42, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '50 days', NOW()),
(91, 'christina.griffin@agency.com', 'Christina', 'Griffin', '+14155552761', 29, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '65 days', NOW()),
(92, 'adam.diaz@innovations.com', 'Adam', 'Diaz', '+14155552762', 47, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '75 days', NOW()),
(93, 'diana.hayes@platforms.com', 'Diana', 'Hayes', '+14155552763', 33, CURRENT_DATE - 16, 'active', NOW() - INTERVAL '85 days', NOW()),
(94, 'henry.myers@networks.com', 'Henry', 'Myers', '+14155552764', 18, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '95 days', NOW()),
(95, 'christine.ford@software.com', 'Christine', 'Ford', '+14155552765', 40, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '45 days', NOW()),
(96, 'carl.hamilton@apps.com', 'Carl', 'Hamilton', '+14155552766', 25, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '55 days', NOW());

-- INACTIVE STATUS (10% = ~12 records)
INSERT INTO analytics.customer_engagement VALUES
(97, 'joyce.graham@design.com', 'Joyce', 'Graham', '+14155552767', 75, CURRENT_DATE - 3, 'inactive', NOW() - INTERVAL '60 days', NOW()),
(98, 'randy.sullivan@creative.com', 'Randy', 'Sullivan', '+14155552768', 82, CURRENT_DATE - 15, 'inactive', NOW() - INTERVAL '70 days', NOW()),
(99, 'jacqueline.wallace@dev.com', 'Jacqueline', 'Wallace', '+14155552769', 68, CURRENT_DATE - 6, 'inactive', NOW() - INTERVAL '80 days', NOW()),
(100, 'phillip.woods@web.com', 'Phillip', 'Woods', '+14155552770', 91, CURRENT_DATE - 9, 'inactive', NOW() - INTERVAL '90 days', NOW()),
(101, 'martha.cole@mobile.com', 'Martha', 'Cole', '+14155552771', 77, CURRENT_DATE - 11, 'inactive', NOW() - INTERVAL '100 days', NOW()),
(102, 'bobby.west@ecommerce.com', 'Bobby', 'West', '+14155552772', 84, CURRENT_DATE - 5, 'inactive', NOW() - INTERVAL '50 days', NOW()),
(103, 'sara.jordan@retail.com', 'Sara', 'Jordan', '+14155552773', 70, CURRENT_DATE - 2, 'inactive', NOW() - INTERVAL '65 days', NOW()),
(104, 'howard.owens@finance.com', 'Howard', 'Owens', '+14155552774', 88, CURRENT_DATE - 14, 'inactive', NOW() - INTERVAL '75 days', NOW()),
(105, 'pamela.reynolds@banking.com', 'Pamela', 'Reynolds', '+14155552775', 73, CURRENT_DATE - 7, 'inactive', NOW() - INTERVAL '85 days', NOW()),
(106, 'douglas.fisher@invest.com', 'Douglas', 'Fisher', '+14155552776', 86, CURRENT_DATE - 10, 'inactive', NOW() - INTERVAL '95 days', NOW()),
(107, 'kathryn.ellis@wealth.com', 'Kathryn', 'Ellis', '+14155552777', 79, CURRENT_DATE - 1, 'inactive', NOW() - INTERVAL '45 days', NOW()),
(108, 'billy.harrison@insurance.com', 'Billy', 'Harrison', '+14155552778', 92, CURRENT_DATE - 13, 'inactive', NOW() - INTERVAL '55 days', NOW());

-- OLD LAST ACTIVITY DATE: > 30 days ago (10% = ~12 records)
INSERT INTO analytics.customer_engagement VALUES
(109, 'virginia.gibson@healthcare.com', 'Virginia', 'Gibson', '+14155552779', 85, CURRENT_DATE - 45, 'active', NOW() - INTERVAL '60 days', NOW()),
(110, 'ernest.mcdonald@medical.com', 'Ernest', 'McDonald', '+14155552780', 78, CURRENT_DATE - 60, 'active', NOW() - INTERVAL '70 days', NOW()),
(111, 'janice.cruz@pharma.com', 'Janice', 'Cruz', '+14155552781', 91, CURRENT_DATE - 50, 'active', NOW() - INTERVAL '80 days', NOW()),
(112, 'carlos.marshall@biotech.com', 'Carlos', 'Marshall', '+14155552782', 74, CURRENT_DATE - 40, 'active', NOW() - INTERVAL '90 days', NOW()),
(113, 'ruby.ortiz@research.com', 'Ruby', 'Ortiz', '+14155552783', 87, CURRENT_DATE - 55, 'active', NOW() - INTERVAL '100 days', NOW()),
(114, 'bruce.gomez@labs.com', 'Bruce', 'Gomez', '+14155552784', 69, CURRENT_DATE - 65, 'active', NOW() - INTERVAL '50 days', NOW()),
(115, 'heather.murray@science.com', 'Heather', 'Murray', '+14155552785', 83, CURRENT_DATE - 35, 'active', NOW() - INTERVAL '65 days', NOW()),
(116, 'craig.freeman@education.com', 'Craig', 'Freeman', '+14155552786', 76, CURRENT_DATE - 70, 'active', NOW() - INTERVAL '75 days', NOW()),
(117, 'marilyn.wells@learning.com', 'Marilyn', 'Wells', '+14155552787', 89, CURRENT_DATE - 48, 'active', NOW() - INTERVAL '85 days', NOW()),
(118, 'willie.webb@training.com', 'Willie', 'Webb', '+14155552788', 72, CURRENT_DATE - 53, 'active', NOW() - INTERVAL '95 days', NOW()),
(119, 'beverly.simpson@courses.com', 'Beverly', 'Simpson', '+14155552789', 84, CURRENT_DATE - 42, 'active', NOW() - INTERVAL '45 days', NOW()),
(120, 'victor.stevens@university.com', 'Victor', 'Stevens', '+14155552790', 80, CURRENT_DATE - 58, 'active', NOW() - INTERVAL '55 days', NOW());

-- DATA QUALITY ISSUES: Invalid emails, engagement scores out of range, duplicates, nulls, bad phone formats
INSERT INTO analytics.customer_engagement VALUES
-- Invalid email formats
(121, 'bademailemail.com', 'Invalid', 'Email1', '+14155552791', 75, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '60 days', NOW()),
(122, 'nodomainemail@', 'Invalid', 'Email2', '+14155552792', 82, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '70 days', NOW()),
(123, '@nodomain.com', 'Invalid', 'Email3', '+14155552793', 68, CURRENT_DATE - 10, 'active', NOW() - INTERVAL '80 days', NOW()),
(124, 'spaces in@email.com', 'Invalid', 'Email4', '+14155552794', 91, CURRENT_DATE - 1, 'active', NOW() - INTERVAL '90 days', NOW()),
-- Engagement scores out of range
(125, 'outsiderange1@email.com', 'OutRange', 'Score1', '+14155552795', 150, CURRENT_DATE - 7, 'active', NOW() - INTERVAL '100 days', NOW()),
(126, 'outsiderange2@email.com', 'OutRange', 'Score2', '+14155552796', -10, CURRENT_DATE - 15, 'active', NOW() - INTERVAL '50 days', NOW()),
(127, 'outsiderange3@email.com', 'OutRange', 'Score3', '+14155552797', 255, CURRENT_DATE - 4, 'active', NOW() - INTERVAL '65 days', NOW()),
-- Duplicate emails (duplicate of record 1)
(128, 'sarah.johnson@email.com', 'Sarah', 'Duplicate', '+14155552798', 88, CURRENT_DATE - 12, 'active', NOW() - INTERVAL '75 days', NOW()),
(129, 'sarah.johnson@email.com', 'Sarah', 'Duplicate2', '+14155552799', 72, CURRENT_DATE - 3, 'active', NOW() - INTERVAL '85 days', NOW()),
-- Null values in required fields
(130, NULL, 'Null', 'Email', '+14155552800', 85, CURRENT_DATE - 8, 'active', NOW() - INTERVAL '95 days', NOW()),
(131, 'nullfirstname@email.com', NULL, 'LastName', '+14155552801', 78, CURRENT_DATE - 6, 'active', NOW() - INTERVAL '45 days', NOW()),
(132, 'nulllastname@email.com', 'FirstName', NULL, '+14155552802', 91, CURRENT_DATE - 11, 'active', NOW() - INTERVAL '55 days', NOW()),
(133, 'nullscore@email.com', 'Null', 'Score', '+14155552803', NULL, CURRENT_DATE - 9, 'active', NOW() - INTERVAL '60 days', NOW()),
-- Phone numbers not in E.164 format
(134, 'badphone1@email.com', 'Bad', 'Phone1', '415-555-2804', 74, CURRENT_DATE - 14, 'active', NOW() - INTERVAL '70 days', NOW()),
(135, 'badphone2@email.com', 'Bad', 'Phone2', '(415) 555-2805', 87, CURRENT_DATE - 5, 'active', NOW() - INTERVAL '80 days', NOW()),
(136, 'badphone3@email.com', 'Bad', 'Phone3', '4155552806', 69, CURRENT_DATE - 2, 'active', NOW() - INTERVAL '90 days', NOW()),
(137, 'badphone4@email.com', 'Bad', 'Phone4', '555-2807', 83, CURRENT_DATE - 13, 'active', NOW() - INTERVAL '100 days', NOW());

-- Summary comments
-- Total records: 137
-- Valid records meeting all criteria: 72 (52%)
-- Low engagement: 24 (17%)
-- Inactive: 12 (9%)
-- Old activity: 12 (9%)
-- Data quality issues: 17 (12%)