#!/usr/bin/env python3

from flask import Flask, request, render_template_string

app = Flask(__name__)

@app.route('/render', methods=['POST'])
def render():
    template_string = request.form.get('template_string', '')
    try:
        rendered = render_template_string(template_string)
        return rendered
    except Exception as e:
        return str(e), 500

@app.route('/', methods=['GET'])
def index():
    return '''
    <html>
        <body>
            <h1>Template Renderer</h1>
            <form method="POST" action="/render">
                <textarea name="template_string" rows="10" cols="50"></textarea><br>
                <input type="submit" value="Render">
            </form>
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)