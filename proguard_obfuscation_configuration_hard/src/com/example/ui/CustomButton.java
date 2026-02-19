package com.example.ui;

import android.content.Context;
import android.graphics.Color;
import android.util.AttributeSet;
import android.widget.Button;

public class CustomButton extends Button {
    
    private int customColor;
    private int borderWidth;
    private boolean isRounded;
    private String buttonStyle;
    
    public CustomButton(Context context) {
        super(context);
        init();
    }
    
    public CustomButton(Context context, AttributeSet attrs) {
        super(context, attrs);
        init();
    }
    
    public CustomButton(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
        init();
    }
    
    private void init() {
        this.customColor = Color.BLUE;
        this.borderWidth = 2;
        this.isRounded = true;
        this.buttonStyle = "default";
        applyDefaultStyle();
    }
    
    public void setCustomStyle(String style) {
        this.buttonStyle = style;
        applyStyle();
    }
    
    public void setCustomColor(int color) {
        this.customColor = color;
        invalidate();
    }
    
    public void setBorderWidth(int width) {
        this.borderWidth = width;
        requestLayout();
    }
    
    private void applyDefaultStyle() {
        setTextColor(Color.WHITE);
        setPadding(20, 10, 20, 10);
    }
    
    private void applyStyle() {
        if ("primary".equals(buttonStyle)) {
            setCustomColor(Color.BLUE);
        } else if ("secondary".equals(buttonStyle)) {
            setCustomColor(Color.GRAY);
        }
    }
}