package com.example.myapp.callbacks;

import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import org.greenrobot.eventbus.Subscribe;
import org.greenrobot.eventbus.ThreadMode;
import retrofit2.http.Body;
import retrofit2.http.GET;
import retrofit2.http.POST;
import retrofit2.Call;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class LibraryCallbacks {

    public static class NetworkCallback {
        public void onSuccess(String response) {
            System.out.println("Network request succeeded: " + response);
        }

        public void onError(Exception e) {
            System.err.println("Network request failed: " + e.getMessage());
            e.printStackTrace();
        }

        public void onProgress(int percent) {
            System.out.println("Progress: " + percent + "%");
        }
    }

    public static class CustomData {
        private String id;
        private Map<String, Object> properties;

        public CustomData() {
            this.properties = new HashMap<>();
        }

        public CustomData(String id, Map<String, Object> properties) {
            this.id = id;
            this.properties = properties;
        }

        public String getId() {
            return id;
        }

        public void setId(String id) {
            this.id = id;
        }

        public Map<String, Object> getProperties() {
            return properties;
        }

        public void setProperties(Map<String, Object> properties) {
            this.properties = properties;
        }
    }

    public static class GsonTypeAdapter extends TypeAdapter<CustomData> {
        @Override
        public void write(JsonWriter out, CustomData value) throws IOException {
            if (value == null) {
                out.nullValue();
                return;
            }
            out.beginObject();
            out.name("id").value(value.getId());
            out.name("properties");
            out.beginObject();
            if (value.getProperties() != null) {
                for (Map.Entry<String, Object> entry : value.getProperties().entrySet()) {
                    out.name(entry.getKey());
                    if (entry.getValue() instanceof String) {
                        out.value((String) entry.getValue());
                    } else if (entry.getValue() instanceof Number) {
                        out.value((Number) entry.getValue());
                    } else if (entry.getValue() instanceof Boolean) {
                        out.value((Boolean) entry.getValue());
                    } else {
                        out.value(entry.getValue().toString());
                    }
                }
            }
            out.endObject();
            out.endObject();
        }

        @Override
        public CustomData read(JsonReader in) throws IOException {
            if (in.peek() == com.google.gson.stream.JsonToken.NULL) {
                in.nextNull();
                return null;
            }
            CustomData data = new CustomData();
            in.beginObject();
            while (in.hasNext()) {
                String name = in.nextName();
                if (name.equals("id")) {
                    data.setId(in.nextString());
                } else if (name.equals("properties")) {
                    Map<String, Object> properties = new HashMap<>();
                    in.beginObject();
                    while (in.hasNext()) {
                        String key = in.nextName();
                        switch (in.peek()) {
                            case STRING:
                                properties.put(key, in.nextString());
                                break;
                            case NUMBER:
                                properties.put(key, in.nextDouble());
                                break;
                            case BOOLEAN:
                                properties.put(key, in.nextBoolean());
                                break;
                            default:
                                in.skipValue();
                        }
                    }
                    in.endObject();
                    data.setProperties(properties);
                } else {
                    in.skipValue();
                }
            }
            in.endObject();
            return data;
        }
    }

    public static class RequestData {
        private String userId;
        private String payload;
        private long timestamp;

        public RequestData(String userId, String payload) {
            this.userId = userId;
            this.payload = payload;
            this.timestamp = System.currentTimeMillis();
        }

        public String getUserId() {
            return userId;
        }

        public void setUserId(String userId) {
            this.userId = userId;
        }

        public String getPayload() {
            return payload;
        }

        public void setPayload(String payload) {
            this.payload = payload;
        }

        public long getTimestamp() {
            return timestamp;
        }

        public void setTimestamp(long timestamp) {
            this.timestamp = timestamp;
        }
    }

    public static class UserData {
        private String username;
        private String email;
        private int age;

        public UserData() {
        }

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getEmail() {
            return email;
        }

        public void setEmail(String email) {
            this.email = email;
        }

        public int getAge() {
            return age;
        }

        public void setAge(int age) {
            this.age = age;
        }
    }

    public interface RetrofitService {
        @GET("/api/user")
        Call<UserData> getUserData();

        @POST("/api/upload")
        Call<Void> uploadData(@Body RequestData data);
    }

    public static class UserEvent {
        private String userId;
        private String action;

        public UserEvent(String userId, String action) {
            this.userId = userId;
            this.action = action;
        }

        public String getUserId() {
            return userId;
        }

        public String getAction() {
            return action;
        }
    }

    public static class NetworkEvent {
        private boolean connected;
        private String networkType;

        public NetworkEvent(boolean connected, String networkType) {
            this.connected = connected;
            this.networkType = networkType;
        }

        public boolean isConnected() {
            return connected;
        }

        public String getNetworkType() {
            return networkType;
        }
    }

    public static class EventBusSubscriber {
        @Subscribe(threadMode = ThreadMode.MAIN)
        public void handleUserEvent(UserEvent event) {
            System.out.println("User event received: " + event.getUserId() + " - " + event.getAction());
        }

        @Subscribe(threadMode = ThreadMode.BACKGROUND)
        public void handleNetworkEvent(NetworkEvent event) {
            System.out.println("Network event: " + (event.isConnected() ? "Connected" : "Disconnected") + " - " + event.getNetworkType());
        }
    }
}