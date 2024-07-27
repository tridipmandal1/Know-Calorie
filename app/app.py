from flask import Flask, request, render_template
from predict import predict_calories

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        img_path = 'temp_image.jpg'
        file.save(img_path)

        calories = predict_calories(img_path)
        return render_template('result.html', calories=calories)
    except Exception as e:
        return render_template('result.html', error=f'An error occurred: {e}')

if __name__ == '__main__':
    app.run(debug=True)
