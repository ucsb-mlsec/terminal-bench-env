package com.example.ui;

import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.widget.Toast;

/**
 * Utility class for managing dialogs throughout the application.
 * Provides static methods for showing common dialog types.
 */
public class DialogManager {

    /**
     * Shows a simple alert dialog with a title and message.
     * @param context The context to use for creating the dialog
     * @param title The dialog title
     * @param message The dialog message
     */
    public static void showAlert(Context context, String title, String message) {
        if (context == null) {
            return;
        }
        
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setTitle(title);
        builder.setMessage(message);
        builder.setPositiveButton("OK", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                dialog.dismiss();
            }
        });
        builder.create().show();
    }

    /**
     * Shows a confirmation dialog with Yes/No buttons.
     * @param context The context to use for creating the dialog
     * @param message The confirmation message
     * @param listener Callback for handling user response
     */
    public static void showConfirmation(Context context, String message, OnConfirmListener listener) {
        if (context == null) {
            return;
        }
        
        AlertDialog.Builder builder = new AlertDialog.Builder(context);
        builder.setMessage(message);
        builder.setPositiveButton("Yes", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (listener != null) {
                    listener.onConfirm(true);
                }
            }
        });
        builder.setNegativeButton("No", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int which) {
                if (listener != null) {
                    listener.onConfirm(false);
                }
            }
        });
        builder.create().show();
    }

    /**
     * Shows an error dialog with a predefined error style.
     * @param context The context to use for creating the dialog
     * @param error The error message to display
     */
    public static void showError(Context context, String error) {
        showAlert(context, "Error", error);
    }

    /**
     * Listener interface for confirmation dialog callbacks.
     */
    public interface OnConfirmListener {
        void onConfirm(boolean confirmed);
    }
}