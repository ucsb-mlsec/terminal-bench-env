from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('form.html')

@app.route('/submit', methods=['POST'])
def submit():
    name = request.form.get('name')
    email = request.form.get('email')
    return jsonify({'status': 'success', 'message': f'Form submitted for {name} ({email})'})

@app.route('/delete', methods=['POST'])
def delete():
    item_id = request.form.get('id')
    return jsonify({'status': 'success', 'message': f'Item {item_id} deleted'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)