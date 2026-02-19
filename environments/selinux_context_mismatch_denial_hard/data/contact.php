<?php
// Contact form processing
$success = false;
$errors = array();

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    // Validate name
    if (empty($_POST['name'])) {
        $errors[] = "Name is required";
    } elseif (strlen($_POST['name']) < 2) {
        $errors[] = "Name must be at least 2 characters";
    }
    
    // Validate email
    if (empty($_POST['email'])) {
        $errors[] = "Email is required";
    } elseif (!filter_var($_POST['email'], FILTER_VALIDATE_EMAIL)) {
        $errors[] = "Invalid email format";
    }
    
    // Validate message
    if (empty($_POST['message'])) {
        $errors[] = "Message is required";
    } elseif (strlen($_POST['message']) < 10) {
        $errors[] = "Message must be at least 10 characters";
    }
    
    // If no errors, process the form
    if (empty($errors)) {
        $success = true;
        // In a real application, you would send email or save to database here
    }
}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Contact Us</title>
</head>
<body>
    <h1>Contact Us</h1>
    
    <?php if ($success): ?>
        <div style="color: green; padding: 10px; border: 1px solid green; margin-bottom: 20px;">
            <strong>Success!</strong> Your message has been sent. We'll get back to you soon.
        </div>
    <?php else: ?>
        <?php if (!empty($errors)): ?>
            <div style="color: red; padding: 10px; border: 1px solid red; margin-bottom: 20px;">
                <strong>Errors:</strong>
                <ul>
                    <?php foreach ($errors as $error): ?>
                        <li><?php echo htmlspecialchars($error); ?></li>
                    <?php endforeach; ?>
                </ul>
            </div>
        <?php endif; ?>
        
        <form method="POST" action="contact.php">
            <p>
                <label for="name">Name:</label><br>
                <input type="text" id="name" name="name" value="<?php echo isset($_POST['name']) ? htmlspecialchars($_POST['name']) : ''; ?>" size="40">
            </p>
            <p>
                <label for="email">Email:</label><br>
                <input type="email" id="email" name="email" value="<?php echo isset($_POST['email']) ? htmlspecialchars($_POST['email']) : ''; ?>" size="40">
            </p>
            <p>
                <label for="message">Message:</label><br>
                <textarea id="message" name="message" rows="6" cols="40"><?php echo isset($_POST['message']) ? htmlspecialchars($_POST['message']) : ''; ?></textarea>
            </p>
            <p>
                <input type="submit" value="Send Message">
            </p>
        </form>
    <?php endif; ?>
</body>
</html>