package com.example.data;

import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.database.Cursor;
import android.content.ContentValues;
import android.content.Context;

public class DatabaseHelper extends SQLiteOpenHelper {
    
    private static final String DATABASE_NAME = "app_database.db";
    private static final int DATABASE_VERSION = 1;
    
    public static final String TABLE_USERS = "users";
    public static final String COLUMN_ID = "id";
    public static final String COLUMN_NAME = "name";
    public static final String COLUMN_EMAIL = "email";
    public static final String COLUMN_CREATED_AT = "created_at";
    
    private static final String CREATE_TABLE_USERS = 
        "CREATE TABLE " + TABLE_USERS + " (" +
        COLUMN_ID + " INTEGER PRIMARY KEY AUTOINCREMENT, " +
        COLUMN_NAME + " TEXT NOT NULL, " +
        COLUMN_EMAIL + " TEXT NOT NULL, " +
        COLUMN_CREATED_AT + " TIMESTAMP DEFAULT CURRENT_TIMESTAMP" +
        ")";
    
    public DatabaseHelper(Context context) {
        super(context, DATABASE_NAME, null, DATABASE_VERSION);
    }
    
    @Override
    public void onCreate(SQLiteDatabase db) {
        db.execSQL(CREATE_TABLE_USERS);
    }
    
    @Override
    public void onUpgrade(SQLiteDatabase db, int oldVersion, int newVersion) {
        db.execSQL("DROP TABLE IF EXISTS " + TABLE_USERS);
        onCreate(db);
    }
    
    public long insertUser(String name, String email) {
        SQLiteDatabase db = this.getWritableDatabase();
        ContentValues values = new ContentValues();
        values.put(COLUMN_NAME, name);
        values.put(COLUMN_EMAIL, email);
        long id = db.insert(TABLE_USERS, null, values);
        db.close();
        return id;
    }
    
    public Cursor getUser(long userId) {
        SQLiteDatabase db = this.getReadableDatabase();
        String[] columns = {COLUMN_ID, COLUMN_NAME, COLUMN_EMAIL, COLUMN_CREATED_AT};
        String selection = COLUMN_ID + " = ?";
        String[] selectionArgs = {String.valueOf(userId)};
        Cursor cursor = db.query(TABLE_USERS, columns, selection, selectionArgs, null, null, null);
        return cursor;
    }
    
    public int deleteUser(long userId) {
        SQLiteDatabase db = this.getWritableDatabase();
        String whereClause = COLUMN_ID + " = ?";
        String[] whereArgs = {String.valueOf(userId)};
        int rowsDeleted = db.delete(TABLE_USERS, whereClause, whereArgs);
        db.close();
        return rowsDeleted;
    }
    
    public Cursor getAllUsers() {
        SQLiteDatabase db = this.getReadableDatabase();
        return db.query(TABLE_USERS, null, null, null, null, null, COLUMN_NAME + " ASC");
    }
}