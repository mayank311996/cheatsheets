from flask import Flask, render_template, request, jsonify

#########################################################################################
app = Flask(__name__)


# To render homepage
@app.route('/', methods=['GET', 'POST'])
def home_page():
    return render_template('index.html')


# This will be called from UI
@app.route('/math', methods=['POST'])
def math_operation():
    if request.method == 'POST':
        operation = request.form['operation']
        num1 = int(request.form['num1'])
        num2 = int(request.form['num2'])
        if operation == 'add':
            r = num1 + num2
            result = 'the sum of ' + str(num1) + ' and ' + str(num2) \
                     + ' is ' + str(r)
        if operation == 'substract':
            r = num1 - num2
            result = 'the difference of ' + str(num1) + ' and ' \
                     + str(num2) + ' is ' + str(r)
         