#!/usr/bin/env python3

from flask import Flask, request, render_template_string
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Feedback Portal</title>
    </head>
    <body>
        <h1>Customer Feedback Portal</h1>
        <form method="POST" action="/submit">
            <label>Name:</label><br>
            <input type="text" name="customer_name" required><br><br>
            
            <label>Email:</label><br>
            <input type="email" name="email" required><br><br>
            
            <label>Feedback:</label><br>
            <textarea name="feedback_text" rows="5" cols="50" required></textarea><br><br>
            
            <input type="submit" value="Submit Feedback">
        </form>
    </body>
    </html>
    '''

@app.route('/submit', methods=['POST'])
def submit():
    customer_name = request.form.get('customer_name', '')
    email = request.form.get('email', '')
    feedback_text = request.form.get('feedback_text', '')
    feedback_date = datetime.now().strftime('%Y-%m-%d')
    
    # VULNERABLE: Directly interpolating user input into template string
    email_template = f'''
    <html>
    <body>
        <h2>Feedback Confirmation</h2>
        <p>Dear {customer_name},</p>
        <p>Thank you for your feedback submitted on {feedback_date}.</p>
        <p>Your feedback: {feedback_text}</p>
        <p>We will review your comments and get back to you at {email}.</p>
        <hr>
        <p>Best regards,<br>Customer Support Team</p>
    </body>
    </html>
    '''
    
    # VULNERABLE: Using render_template_string on user-controlled data
    rendered = render_template_string(email_template)
    
    return rendered

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)