<?php

require_once __DIR__ . '/src/User.php';

use App\User;

echo "Testing User Mass Assignment Protection\n";
echo "========================================\n\n";

$allTestsPassed = true;

// Test 1: Attempt to mass assign allowed profile fields
echo "Test 1: Mass assign allowed profile fields\n";
echo "-------------------------------------------\n";
try {
    $user1 = new User();
    $allowedData = [
        'name' => 'John Doe',
        'email' => 'john@example.com',
        'bio' => 'Software developer',
        'avatar_url' => 'https://example.com/avatar.jpg',
        'timezone' => 'America/New_York'
    ];
    
    $user1->fill($allowedData);
    
    // Verify all allowed fields are set
    $allFieldsSet = true;
    foreach ($allowedData as $field => $value) {
        if (!isset($user1->$field) || $user1->$field !== $value) {
            echo "FAIL: Field '$field' was not set correctly\n";
            $allFieldsSet = false;
            $allTestsPassed = false;
        }
    }
    
    if ($allFieldsSet) {
        echo "PASS: All allowed profile fields were set correctly\n";
    }
} catch (Exception $e) {
    echo "FAIL: Exception occurred - " . $e->getMessage() . "\n";
    $allTestsPassed = false;
}
echo "\n";

// Test 2: Attempt to mass assign protected security fields
echo "Test 2: Mass assign protected security fields\n";
echo "----------------------------------------------\n";
try {
    $user2 = new User();
    $protectedData = [
        'password' => 'hashed_password',
        'remember_token' => 'some_token',
        'is_admin' => true,
        'email_verified_at' => '2024-01-01 00:00:00'
    ];
    
    $user2->fill($protectedData);
    
    // Verify none of the protected fields are set
    $allFieldsBlocked = true;
    foreach ($protectedData as $field => $value) {
        if (isset($user2->$field)) {
            echo "FAIL: Protected field '$field' was incorrectly set\n";
            $allFieldsBlocked = false;
            $allTestsPassed = false;
        }
    }
    
    if ($allFieldsBlocked) {
        echo "PASS: All protected fields were blocked correctly\n";
    }
} catch (Exception $e) {
    echo "FAIL: Exception occurred - " . $e->getMessage() . "\n";
    $allTestsPassed = false;
}
echo "\n";

// Test 3: Mixed assignment (both allowed and blocked fields)
echo "Test 3: Mixed assignment (allowed and blocked fields)\n";
echo "------------------------------------------------------\n";
try {
    $user3 = new User();
    $mixedData = [
        'name' => 'Jane Smith',
        'email' => 'jane@example.com',
        'password' => 'should_not_be_set',
        'is_admin' => true
    ];
    
    $user3->fill($mixedData);
    
    // Verify allowed fields are set
    $allowedFieldsCorrect = true;
    if (!isset($user3->name) || $user3->name !== 'Jane Smith') {
        echo "FAIL: Allowed field 'name' was not set\n";
        $allowedFieldsCorrect = false;
        $allTestsPassed = false;
    }
    if (!isset($user3->email) || $user3->email !== 'jane@example.com') {
        echo "FAIL: Allowed field 'email' was not set\n";
        $allowedFieldsCorrect = false;
        $allTestsPassed = false;
    }
    
    // Verify blocked fields are not set
    if (isset($user3->password)) {
        echo "FAIL: Protected field 'password' was incorrectly set\n";
        $allowedFieldsCorrect = false;
        $allTestsPassed = false;
    }
    if (isset($user3->is_admin)) {
        echo "FAIL: Protected field 'is_admin' was incorrectly set\n";
        $allowedFieldsCorrect = false;
        $allTestsPassed = false;
    }
    
    if ($allowedFieldsCorrect) {
        echo "PASS: Mixed assignment filtered correctly\n";
    }
} catch (Exception $e) {
    echo "FAIL: Exception occurred - " . $e->getMessage() . "\n";
    $allTestsPassed = false;
}
echo "\n";

// Final summary
echo "========================================\n";
if ($allTestsPassed) {
    echo "ALL TESTS PASSED!\n";
    exit(0);
} else {
    echo "SOME TESTS FAILED!\n";
    exit(1);
}