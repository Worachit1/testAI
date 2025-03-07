from flask import Flask, request, render_template
import pandas as pd
import joblib
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# LINE Bot configuration
LINE_CHANNEL_ACCESS_TOKEN = 'YOUR_CHANNEL_ACCESS_TOKEN'
LINE_CHANNEL_SECRET = 'YOUR_CHANNEL_SECRET'

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# Load model and data
model = joblib.load('Ensemble_Model.pkl')
employee_data = pd.read_csv('data.csv')

@app.route('/')
def home():
    # Convert employee data to HTML-friendly format
    employee_data_html = employee_data.to_dict(orient='records')
    return render_template('index.html', employee_data=employee_data_html)

@app.route('/predict', methods=['POST'])
def predict():
    # Receive input from the form
    age = int(request.form['age'])
    length_of_service = int(request.form['length_of_service'])
    gender = int(request.form['gender'])
    marital_status = int(request.form['marital_status'])
    salary = float(request.form['salary'])

    # Prepare input data for prediction
    input_data = pd.DataFrame([[age, length_of_service, salary, gender, marital_status]],
                              columns=['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status'])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = 'Still Employed' if prediction == 1 else 'Resigned'

    return render_template(
        'index.html',
        prediction_text=f'Employee Status: {result}',
        employee_data=employee_data.to_dict(orient='records'),
        age=age,
        length_of_service=length_of_service,
        salary=salary,
        gender=gender,
        marital_status=marital_status
    )

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        return 'Invalid signature', 400

    return 'OK', 200

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    # Example: Parse the user's message and perform prediction
    user_message = event.message.text
    if user_message.lower() == 'predict':
        # Assuming you send the parameters (age, salary, etc.) in a predefined format
        # You can parse the input parameters here
        age = 30  # Example static value
        length_of_service = 5  # Example static value
        gender = 1  # Example static value
        marital_status = 1  # Example static value
        salary = 50000.0  # Example static value

        input_data = pd.DataFrame([[age, length_of_service, salary, gender, marital_status]],
                                  columns=['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status'])
        
        prediction = model.predict(input_data)[0]
        result = 'Still Employed' if prediction == 1 else 'Resigned'

        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f'Prediction: Employee Status: {result}')
        )

if __name__ == "__main__":
    app.run(debug=True)
