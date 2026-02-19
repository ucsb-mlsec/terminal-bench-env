package com.bankingapp.security;

import android.util.Log;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.security.cert.CertificateEncodingException;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLSession;
import javax.net.ssl.X509TrustManager;

/* compiled from: CertificateValidator.java */
public class CertificateValidator implements X509TrustManager, HostnameVerifier {
    private static final String TAG = "CertificateValidator";
    private static final String PRIMARY_PIN = "sha256/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
    private static final String BACKUP_PIN = "sha256/BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB=";
    private static final String LEGACY_PIN = "sha256/CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC=";
    private static final String EXPECTED_HOSTNAME = "api.securebank.com";
    private static CertificateValidator instance;
    private boolean pinningEnabled = true;

    /* compiled from: CertificateValidator.java */
    private CertificateValidator() {
        Log.d(TAG, "CertificateValidator initialized");
    }

    public static synchronized CertificateValidator getInstance() {
        CertificateValidator certificateValidator;
        synchronized (CertificateValidator.class) {
            if (instance == null) {
                instance = new CertificateValidator();
            }
            certificateValidator = instance;
        }
        return certificateValidator;
    }

    @Override
    public void checkClientTrusted(X509Certificate[] chain, String authType) throws CertificateException {
        Log.d(TAG, "checkClientTrusted called");
    }

    @Override
    public void checkServerTrusted(X509Certificate[] chain, String authType) throws CertificateException {
        Log.d(TAG, "checkServerTrusted called with authType: " + authType);
        if (chain == null || chain.length == 0) {
            throw new CertificateException("Certificate chain is empty");
        }
        
        if (!validateCertificateChain(chain)) {
            throw new CertificateException("Certificate chain validation failed");
        }

        if (this.pinningEnabled && !checkCertificatePins(chain)) {
            Log.e(TAG, "Certificate pinning validation failed");
            throw new CertificateException("Certificate pinning validation failed");
        }
        
        Log.d(TAG, "Server certificate trusted");
    }

    @Override
    public X509Certificate[] getAcceptedIssuers() {
        return new X509Certificate[0];
    }

    @Override
    public boolean verify(String hostname, SSLSession session) {
        Log.d(TAG, "Hostname verification for: " + hostname);
        return verifyHostname(hostname);
    }

    public boolean verifyHostname(String hostname) {
        if (hostname == null || hostname.isEmpty()) {
            Log.e(TAG, "Hostname is null or empty");
            return false;
        }

        if (!hostname.equals(EXPECTED_HOSTNAME)) {
            Log.e(TAG, "Hostname mismatch. Expected: " + EXPECTED_HOSTNAME + ", Got: " + hostname);
            return false;
        }

        Log.d(TAG, "Hostname verification successful");
        return true;
    }

    public boolean checkCertificatePins(X509Certificate[] chain) {
        if (chain == null || chain.length == 0) {
            Log.e(TAG, "Certificate chain is null or empty");
            return false;
        }

        for (X509Certificate cert : chain) {
            try {
                String pin = calculatePin(cert);
                Log.d(TAG, "Calculated pin: " + pin);
                
                if (pin.equals(PRIMARY_PIN) || pin.equals(BACKUP_PIN) || pin.equals(LEGACY_PIN)) {
                    Log.d(TAG, "Certificate pin matched");
                    return true;
                }
            } catch (CertificateEncodingException | NoSuchAlgorithmException e) {
                Log.e(TAG, "Error calculating certificate pin", e);
            }
        }

        Log.e(TAG, "No matching certificate pins found");
        return false;
    }

    public boolean validateCertificateChain(X509Certificate[] chain) {
        if (chain == null || chain.length == 0) {
            Log.e(TAG, "Invalid certificate chain");
            return false;
        }

        for (X509Certificate cert : chain) {
            try {
                cert.checkValidity();
                Log.d(TAG, "Certificate is valid: " + cert.getSubjectDN());
            } catch (Exception e) {
                Log.e(TAG, "Certificate validation failed", e);
                return false;
            }
        }

        Log.d(TAG, "Certificate chain validation successful");
        return true;
    }

    private String calculatePin(X509Certificate certificate) throws CertificateEncodingException, NoSuchAlgorithmException {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] publicKeyBytes = certificate.getPublicKey().getEncoded();
        byte[] hash = digest.digest(publicKeyBytes);
        return "sha256/" + android.util.Base64.encodeToString(hash, android.util.Base64.NO_WRAP);
    }

    public boolean isPinningEnabled() {
        return this.pinningEnabled;
    }

    public void setPinningEnabled(boolean enabled) {
        Log.d(TAG, "Setting pinning enabled: " + enabled);
        this.pinningEnabled = enabled;
    }

    public boolean validateConnection(X509Certificate[] chain, String hostname) {
        if (!verifyHostname(hostname)) {
            Log.e(TAG, "Connection validation failed: hostname verification failed");
            return false;
        }

        if (!validateCertificateChain(chain)) {
            Log.e(TAG, "Connection validation failed: certificate chain invalid");
            return false;
        }

        if (this.pinningEnabled && !checkCertificatePins(chain)) {
            Log.e(TAG, "Connection validation failed: certificate pinning failed");
            return false;
        }

        Log.d(TAG, "Connection validation successful");
        return true;
    }
}