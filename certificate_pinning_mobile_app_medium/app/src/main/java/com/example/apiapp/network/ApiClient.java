package com.example.apiapp.network;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.converter.gson.GsonConverterFactory;

import javax.net.ssl.SSLContext;
import javax.net.ssl.SSLSocketFactory;
import javax.net.ssl.TrustManager;
import javax.net.ssl.X509TrustManager;
import java.security.cert.CertificateException;
import java.security.cert.X509Certificate;
import java.util.concurrent.TimeUnit;

/**
 * ApiClient class for configuring Retrofit network communication
 * 
 * WARNING: This implementation is INSECURE and vulnerable to MITM attacks!
 * It accepts ANY SSL certificate without validation.
 * TODO: Implement certificate pinning to secure communications
 */
public class ApiClient {

    private static Retrofit retrofit = null;
    private static final String BASE_URL = "https://api.example.com/";

    /**
     * Returns a configured Retrofit instance
     * 
     * SECURITY ISSUE: This method creates an OkHttpClient that trusts all certificates
     * without validation, making the app vulnerable to man-in-the-middle attacks.
     * 
     * @return Retrofit instance configured with insecure OkHttpClient
     */
    public static Retrofit getRetrofitInstance() {
        if (retrofit == null) {
            // Create insecure OkHttpClient that accepts all certificates
            OkHttpClient okHttpClient = getUnsafeOkHttpClient();
            
            retrofit = new Retrofit.Builder()
                    .baseUrl(BASE_URL)
                    .client(okHttpClient)
                    .addConverterFactory(GsonConverterFactory.create())
                    .build();
        }
        return retrofit;
    }

    /**
     * Creates an UNSAFE OkHttpClient that accepts all SSL certificates
     * 
     * CRITICAL SECURITY VULNERABILITY: This implementation bypasses all SSL certificate
     * validation, allowing attackers to perform man-in-the-middle attacks.
     * 
     * @return OkHttpClient configured to trust all certificates
     */
    private static OkHttpClient getUnsafeOkHttpClient() {
        try {
            // Create a trust manager that does NOT validate certificate chains
            final TrustManager[] trustAllCerts = new TrustManager[] {
                new X509TrustManager() {
                    @Override
                    public void checkClientTrusted(X509Certificate[] chain, String authType) 
                            throws CertificateException {
                        // Accept all client certificates - INSECURE!
                    }

                    @Override
                    public void checkServerTrusted(X509Certificate[] chain, String authType) 
                            throws CertificateException {
                        // Accept all server certificates - INSECURE!
                        // This is the critical vulnerability - no validation performed
                    }

                    @Override
                    public X509Certificate[] getAcceptedIssuers() {
                        return new X509Certificate[]{};
                    }
                }
            };

            // Install the all-trusting trust manager
            final SSLContext sslContext = SSLContext.getInstance("SSL");
            sslContext.init(null, trustAllCerts, new java.security.SecureRandom());
            
            // Create an SSL socket factory with our all-trusting manager
            final SSLSocketFactory sslSocketFactory = sslContext.getSocketFactory();

            OkHttpClient.Builder builder = new OkHttpClient.Builder();
            builder.sslSocketFactory(sslSocketFactory, (X509TrustManager)trustAllCerts[0]);
            builder.hostnameVerifier((hostname, session) -> true); // Accept all hostnames - INSECURE!
            builder.connectTimeout(30, TimeUnit.SECONDS);
            builder.readTimeout(30, TimeUnit.SECONDS);
            builder.writeTimeout(30, TimeUnit.SECONDS);

            return builder.build();
            
        } catch (Exception e) {
            throw new RuntimeException("Failed to create unsafe OkHttpClient", e);
        }
    }

    /**
     * Resets the Retrofit instance (useful for testing or reconfiguration)
     */
    public static void resetInstance() {
        retrofit = null;
    }
}