#!/usr/bin/env python
from celery import Celery
from PIL import Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import io
import requests
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

app = Celery('tasks', broker='redis://localhost:6379/0')

# Database configuration
engine = create_engine('postgresql://user:password@localhost/orders_db')
Session = sessionmaker(bind=engine)

# Global cache that will cause memory leak
PROCESSED_IMAGES = []


@app.task
def process_order_confirmation(order_id, customer_email, order_details):
    """Process order confirmation and store in database"""
    session = Session()
    try:
        # Query order data
        order_data = {
            'order_id': order_id,
            'customer_email': customer_email,
            'details': order_details,
            'status': 'confirmed'
        }
        
        # Format confirmation message
        confirmation_message = f"Order {order_id} confirmed for {customer_email}"
        
        # Simulate database operation
        # In real code, this would insert/update database records
        
        return {'status': 'success', 'message': confirmation_message}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}
    finally:
        session.close()


@app.task
def process_product_image(image_url, product_id, sizes):
    """Download and process product images in multiple sizes"""
    try:
        # Download image from URL
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        
        # Load image with PIL
        image_data = io.BytesIO(response.content)
        original_image = Image.open(image_data)
        
        # Process image for each size
        processed_count = 0
        for size in sizes:
            # Create thumbnail
            img_copy = original_image.copy()
            img_copy.thumbnail((size, size), Image.Resampling.LANCZOS)
            
            # Convert to bytes
            output = io.BytesIO()
            img_copy.save(output, format=original_image.format or 'JPEG')
            image_bytes = output.getvalue()
            
            # Store processed image data in global list - MEMORY LEAK!
            # This list grows indefinitely with each task execution
            PROCESSED_IMAGES.append({
                'product_id': product_id,
                'size': size,
                'data': image_bytes,
                'image_object': img_copy
            })
            
            processed_count += 1
            
            # In real application, would save to storage/CDN here
            # storage.save(f"product_{product_id}_{size}.jpg", image_bytes)
        
        return {
            'status': 'success',
            'product_id': product_id,
            'processed_sizes': processed_count,
            'cache_size': len(PROCESSED_IMAGES)
        }
    except requests.RequestException as e:
        return {'status': 'error', 'message': f'Failed to download image: {str(e)}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Image processing failed: {str(e)}'}


@app.task
def send_customer_email(customer_email, subject, message_body):
    """Send email notification to customer"""
    smtp_server = 'smtp.example.com'
    smtp_port = 587
    sender_email = 'noreply@example.com'
    sender_password = 'password123'
    
    try:
        # Create SMTP connection
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Compose email
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = customer_email
        
        # Add message body
        text_part = MIMEText(message_body, 'plain')
        msg.attach(text_part)
        
        # Send email
        server.send_message(msg)
        
        return {'status': 'success', 'recipient': customer_email}
    except smtplib.SMTPException as e:
        return {'status': 'error', 'message': f'SMTP error: {str(e)}'}
    except Exception as e:
        return {'status': 'error', 'message': f'Email sending failed: {str(e)}'}
    finally:
        try:
            server.quit()
        except:
            pass
```

Now I'll analyze the code and create the leak report:

```python
#!/usr/bin/env python

# Analyzing the tasks.py file for memory leaks

# Task 1: process_order_confirmation
# - Uses session with proper cleanup in finally block
# - No global state accumulation
# - No leak

# Task 2: process_product_image
# - Uses PROCESSED_IMAGES global list at module level
# - Appends image data and PIL Image objects to this list on every execution
# - List is never cleared or bounded
# - MEMORY LEAK FOUND

# Task 3: send_customer_email
# - SMTP connection properly closed in finally block
# - No global state accumulation
# - No leak

report = """TASK: process_product_image
CAUSE: Processed image data and PIL Image objects are appended to a module-level PROCESSED_IMAGES list that grows indefinitely with each task execution"""

with open('/tmp/leak_report.txt', 'w') as f:
    f.write(report)