from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('Ensemble_Model.pkl')

# โหลดข้อมูลจาก CSV
employee_data = pd.read_csv('data.csv')

@app.route('/')
def home():
    # แปลงข้อมูลพนักงานจาก DataFrame เป็นรูปแบบ dictionary เพื่อแสดงใน HTML
    employee_data_html = employee_data.to_dict(orient='records')
    return render_template('index.html', employee_data=employee_data_html)

@app.route('/predict', methods=['POST'])
def predict():
    # รับค่าจากฟอร์ม
    age = int(request.form['age'])
    length_of_service = int(request.form['length_of_service'])
    gender = int(request.form['gender'])  # 0: Male, 1: Female
    marital_status = int(request.form['marital_status'])  # 0: Single, 1: Married
    salary = float(request.form['salary'])

    # สร้าง DataFrame สำหรับข้อมูลที่กรอก
    input_data = pd.DataFrame([[age, length_of_service, salary, gender, marital_status]],
                              columns=['Age', 'Length_of_Service', 'Salary', 'Gender', 'Marital_Status'])

    # ทำนายผลลัพธ์
    prediction = model.predict(input_data)[0]

    # ส่งผลลัพธ์กลับไปยังหน้าเว็บ
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

if __name__ == "__main__":
    app.run(debug=True)