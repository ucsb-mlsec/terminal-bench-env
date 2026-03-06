package com.example.app;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.Toast;
import com.example.utils.ValidationHelper;
import com.example.auth.AuthService;

public class LoginActivity extends Activity {
    
    private EditText usernameField;
    private EditText passwordField;
    private Button loginButton;
    private AuthService authService;
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_login);
        
        usernameField = findViewById(R.id.username);
        passwordField = findViewById(R.id.password);
        loginButton = findViewById(R.id.login_button);
        authService = new AuthService();
        
        loginButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                validateLogin();
            }
        });
    }
    
    private void validateLogin() {
        String username = usernameField.getText().toString();
        String password = passwordField.getText().toString();
        
        if (ValidationHelper.isValidUsername(username) && 
            ValidationHelper.isValidPassword(password)) {
            
            boolean success = authService.authenticate(username, password);
            if (success) {
                onLoginSuccess();
            } else {
                onLoginFailure();
            }
        } else {
            Toast.makeText(this, "Invalid credentials format", Toast.LENGTH_SHORT).show();
        }
    }
    
    private void onLoginSuccess() {
        Toast.makeText(this, "Login successful", Toast.LENGTH_SHORT).show();
        finish();
    }
    
    private void onLoginFailure() {
        Toast.makeText(this, "Login failed", Toast.LENGTH_SHORT).show();
        passwordField.setText("");
    }
}