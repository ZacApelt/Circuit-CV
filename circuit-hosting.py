from flask import Flask, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/<filename>')
def serve_circuit(filename):
    return send_file(filename)

if __name__ == '__main__':
    app.run(port=7000)
