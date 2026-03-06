<?php
// About Us Page
$page_title = 'About Us';
$company_name = 'WebTech Solutions';
$founded_year = 2015;
$current_year = date('Y');
$years_in_business = $current_year - $founded_year;
$employee_count = 45;
$mission = 'Delivering innovative web solutions that empower businesses to succeed in the digital age';
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title><?php echo $page_title . ' - ' . $company_name; ?></title>
</head>
<body>
    <header>
        <h1><?php echo $company_name; ?></h1>
        <h2><?php echo $page_title; ?></h2>
    </header>
    
    <main>
        <section>
            <h3>Our Story</h3>
            <p>Founded in <?php echo $founded_year; ?>, <?php echo $company_name; ?> has been providing cutting-edge web development and digital solutions for <?php echo $years_in_business; ?> years. What started as a small team of passionate developers has grown into a thriving company with <?php echo $employee_count; ?> dedicated professionals.</p>
            
            <p>Our mission is simple: <em><?php echo $mission; ?></em></p>
        </section>
        
        <section>
            <h3>What We Do</h3>
            <p>We specialize in custom web application development, e-commerce solutions, and digital transformation consulting. Our team works closely with clients to understand their unique challenges and deliver tailored solutions that drive real business results.</p>
            
            <p>From startups to enterprise organizations, we've helped hundreds of companies establish their online presence and optimize their digital operations.</p>
        </section>
        
        <section>
            <h3>Our Values</h3>
            <p>Innovation, integrity, and customer satisfaction are at the core of everything we do. We believe in building long-term partnerships with our clients and delivering solutions that exceed expectations.</p>
        </section>
        
        <footer>
            <p>Last updated: <?php echo date('F j, Y'); ?></p>
            <p>&copy; <?php echo $current_year . ' ' . $company_name; ?>. All rights reserved.</p>
        </footer>
    </main>
</body>
</html>