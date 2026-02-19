package auth

import (
	"errors"
	"testing"

	"github.com/golang/mock/gomock"
)

func TestValidateToken(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockAuth := NewMockAuthenticator(ctrl)

	// Test valid token
	validToken := "valid-token-123"
	mockAuth.EXPECT().ValidateToken(validToken).Return(true, nil)

	result, err := mockAuth.ValidateToken(validToken)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if !result {
		t.Errorf("Expected token to be valid")
	}

	// Test invalid token
	invalidToken := "invalid-token-456"
	mockAuth.EXPECT().ValidateToken(invalidToken).Return(false, nil)

	result, err = mockAuth.ValidateToken(invalidToken)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if result {
		t.Errorf("Expected token to be invalid")
	}

	// Test token validation error
	errorToken := "error-token-789"
	mockAuth.EXPECT().ValidateToken(errorToken).Return(false, errors.New("validation error"))

	result, err = mockAuth.ValidateToken(errorToken)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	if result {
		t.Errorf("Expected token validation to fail")
	}
}

func TestRefreshToken(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockAuth := NewMockAuthenticator(ctrl)

	// Test successful token refresh
	userID := "user-123"
	newToken := "new-token-abc-xyz"
	mockAuth.EXPECT().RefreshToken(userID).Return(newToken, nil)

	token, err := mockAuth.RefreshToken(userID)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
	if token != newToken {
		t.Errorf("Expected token %s, got %s", newToken, token)
	}

	// Test token refresh failure
	invalidUserID := "invalid-user-456"
	mockAuth.EXPECT().RefreshToken(invalidUserID).Return("", errors.New("user not found"))

	token, err = mockAuth.RefreshToken(invalidUserID)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
	if token != "" {
		t.Errorf("Expected empty token, got %s", token)
	}
}

func TestRevokeToken(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockAuth := NewMockAuthenticator(ctrl)

	// Test successful token revocation
	validToken := "token-to-revoke-123"
	mockAuth.EXPECT().RevokeToken(validToken).Return(nil)

	err := mockAuth.RevokeToken(validToken)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}

	// Test token revocation failure
	invalidToken := "invalid-token-456"
	mockAuth.EXPECT().RevokeToken(invalidToken).Return(errors.New("token not found"))

	err = mockAuth.RevokeToken(invalidToken)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}

	// Test token already revoked
	revokedToken := "already-revoked-789"
	mockAuth.EXPECT().RevokeToken(revokedToken).Return(errors.New("token already revoked"))

	err = mockAuth.RevokeToken(revokedToken)
	if err == nil {
		t.Errorf("Expected error, got nil")
	}
}