from flask import Flask, request, jsonify
from devtools.generate_image import save_base64_image
from main import process_image

app = Flask(__name__)

@app.route('/run-script', methods=['POST'])
def run_script():
    # Get JSON data from the request body
    data = request.json.get('data', 'No data provided')
    
    # Split the data on commas and get the last element
    last_element = data.split(',')[-1]
    
    # Save the last element to a file
    with open('outputs/image_string.txt', 'w') as file:
        file.write(last_element)

    save_base64_image('outputs/image_string.txt', 'outputs/chrome_image.png')
    process_image(image_path='outputs/chrome_image.png')
    
    return jsonify({'message': 'Image processed successfully'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)