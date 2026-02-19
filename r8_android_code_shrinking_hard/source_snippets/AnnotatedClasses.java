package com.example.myapp.annotated;

import androidx.room.Entity;
import androidx.room.Table;
import androidx.room.PrimaryKey;
import androidx.room.ColumnInfo;
import androidx.room.Ignore;
import com.google.gson.annotations.SerializedName;
import javax.inject.Inject;
import androidx.work.Worker;
import androidx.work.WorkerParameters;
import androidx.annotation.WorkerThread;
import android.content.Context;
import androidx.annotation.NonNull;

@Entity
@Table(name = "users")
class DatabaseEntity {
    @PrimaryKey(autoGenerate = true)
    private int id;
    
    @ColumnInfo(name = "user_name")
    private String userName;
    
    @ColumnInfo(name = "email")
    private String email;
    
    @ColumnInfo(name = "created_at")
    private long createdAt;
    
    @Ignore
    private transient String tempData;
    
    public DatabaseEntity() {
    }
    
    public DatabaseEntity(String userName, String email) {
        this.userName = userName;
        this.email = email;
        this.createdAt = System.currentTimeMillis();
    }
    
    public int getId() {
        return id;
    }
    
    public void setId(int id) {
        this.id = id;
    }
    
    public String getUserName() {
        return userName;
    }
    
    public void setUserName(String userName) {
        this.userName = userName;
    }
    
    public String getEmail() {
        return email;
    }
    
    public void setEmail(String email) {
        this.email = email;
    }
    
    public long getCreatedAt() {
        return createdAt;
    }
    
    public void setCreatedAt(long createdAt) {
        this.createdAt = createdAt;
    }
    
    public String getTempData() {
        return tempData;
    }
    
    public void setTempData(String tempData) {
        this.tempData = tempData;
    }
}

class RestApiModel {
    @SerializedName("user_id")
    private String userId;
    
    @SerializedName("user_name")
    private String userName;
    
    @SerializedName("email_address")
    private String emailAddress;
    
    @SerializedName("profile_image_url")
    private String profileImageUrl;
    
    @SerializedName("is_verified")
    private boolean isVerified;
    
    @SerializedName("account_type")
    private String accountType;
    
    public RestApiModel() {
    }
    
    public String getUserId() {
        return userId;
    }
    
    public void setUserId(String userId) {
        this.userId = userId;
    }
    
    public String getUserName() {
        return userName;
    }
    
    public void setUserName(String userName) {
        this.userName = userName;
    }
    
    public String getEmailAddress() {
        return emailAddress;
    }
    
    public void setEmailAddress(String emailAddress) {
        this.emailAddress = emailAddress;
    }
    
    public String getProfileImageUrl() {
        return profileImageUrl;
    }
    
    public void setProfileImageUrl(String profileImageUrl) {
        this.profileImageUrl = profileImageUrl;
    }
    
    public boolean isVerified() {
        return isVerified;
    }
    
    public void setVerified(boolean verified) {
        isVerified = verified;
    }
    
    public String getAccountType() {
        return accountType;
    }
    
    public void setAccountType(String accountType) {
        this.accountType = accountType;
    }
}

class ViewModelClass {
    private final UserRepository userRepository;
    private final AnalyticsService analyticsService;
    private final PreferencesManager preferencesManager;
    
    @Inject
    public ViewModelClass(UserRepository userRepository, 
                         AnalyticsService analyticsService,
                         PreferencesManager preferencesManager) {
        this.userRepository = userRepository;
        this.analyticsService = analyticsService;
        this.preferencesManager = preferencesManager;
    }
    
    public UserRepository getUserRepository() {
        return userRepository;
    }
    
    public AnalyticsService getAnalyticsService() {
        return analyticsService;
    }
    
    public PreferencesManager getPreferencesManager() {
        return preferencesManager;
    }
    
    public void loadUserData(String userId) {
        analyticsService.logEvent("load_user_data", userId);
        userRepository.fetchUser(userId);
    }
    
    interface UserRepository {
        void fetchUser(String userId);
    }
    
    interface AnalyticsService {
        void logEvent(String eventName, String param);
    }
    
    interface PreferencesManager {
        void savePreference(String key, String value);
    }
}

class WorkerClass extends Worker {
    private static final String TAG = "WorkerClass";
    
    public WorkerClass(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }
    
    @NonNull
    @Override
    @WorkerThread
    public Result doWork() {
        try {
            String inputData = getInputData().getString("data_key");
            
            performBackgroundTask(inputData);
            
            return Result.success();
        } catch (Exception e) {
            return Result.retry();
        }
    }
    
    @WorkerThread
    private void performBackgroundTask(String data) {
        try {
            Thread.sleep(1000);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}