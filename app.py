from flask import Flask, render_template, redirect, url_for, request, flash, jsonify, session
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from flask_sqlalchemy import SQLAlchemy
from forms import LoginForm, RegisterForm
from models import db, User, Chat
from openai import OpenAI
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import os
import random

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'supersecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

with app.app_context():
    db.create_all()

@app.route('/', methods=['GET', 'POST'])
@login_required
def chat():
    if request.method == 'POST':
        user_input = request.form['message'].strip().lower()

        if user_input in ['hi', 'hello', 'start', 'menu', 'help']:
            session['state'] = 'main_menu'
            return jsonify({"response": """
👋 Welcome to the Python + AI Tutor Bot!

Choose a topic to explore:

1. 🐍 Python Basics  
2. ⚙️ Advanced Python  
3. 🤖 AI Topics  
4. 🎲 Random Questions

💡 Type the number (1–4) to continue.
💡 You can type `menu` anytime to return here.
"""})

        if user_input == 'menu':
            session['state'] = 'main_menu'
            return jsonify({"response": """
🔄 You're back at the main menu!

1. 🐍 Python Basics  
2. ⚙️ Advanced Python  
3. 🤖 AI Topics  
4. 🎲 Random Questions

Type 1, 2, 3, or 4 to begin again.
"""})

        if session.get('state') == 'main_menu':
            if user_input == '1':
                session['state'] = 'python_basics'
                return jsonify({"response": """
📘 Python Basics - Choose a sub-topic:

1. Variables  
2. Loops  
3. Functions  
4. Lists  
5. Dictionaries

💡 Type `menu` to go back.
"""})
            elif user_input == '2':
                session['state'] = 'advanced_python'
                return jsonify({"response": """
⚙️ Advanced Python - Choose a sub-topic:

1. List Comprehensions  
2. Decorators  
3. Generators

💡 Type `menu` to go back.
"""})
            elif user_input == '3':
                session['state'] = 'ai_topics'
                return jsonify({"response": """
🤖 AI Topics - Choose a sub-topic:

1. What is AI?  
2. What is Machine Learning?  
3. What is a Chatbot?

💡 Type `menu` to go back.
"""})
            elif user_input == '4':
                from questions import questions
                selected = random.choice(questions)
                return jsonify({
                    "response": f"📘 {selected['question']}\n\n💡 {selected['answer']}\n\nType `menu` to choose again."
                })

        if session.get('state') == 'python_basics':
            if user_input == '1':
                return jsonify({"response": """
🔢 **Variables in Python**  
Variables store data values.

```python
x = 10  
name = "Alice"  
print(x, name)
```

💡 Type `menu` to go back.
"""})
            elif user_input == '2':
                return jsonify({"response": """
🔁 **Loops in Python**  
Use `for` or `while` to repeat tasks.

```python
for i in range(5):  
    print(i)
```

💡 Type `menu` to go back.
"""})
            elif user_input == '3':
                return jsonify({"response": """
📘 **Functions in Python**  
Functions group code blocks.

```python
def greet(name):  
    return f"Hello {name}"
```

💡 Type `menu` to go back.
"""})
            elif user_input == '4':
                return jsonify({"response": """
📋 **Lists in Python**  
Lists are ordered and changeable.

```python
fruits = ['apple', 'banana']  
fruits.append('mango')  
print(fruits)
```

💡 Type `menu` to go back.
"""})
            elif user_input == '5':
                return jsonify({"response": """
🧾 **Dictionaries in Python**  
They store key-value pairs.

```python
person = {"name": "Ali", "age": 22}  
print(person["name"])
```

💡 Type `menu` to go back.
"""})

        if session.get('state') == 'advanced_python':
            if user_input == '1':
                return jsonify({"response": """
🧠 **List Comprehensions**  
A concise way to create lists.

```python
squares = [x*x for x in range(5)]
print(squares)
```

💡 Type `menu` to go back.
"""})
            elif user_input == '2':
                return jsonify({"response": """
🎁 **Decorators**  
Used to modify functions.

```python
def decorator(func):  
    def wrapper():  
        print("Before")  
        func()  
        print("After")  
    return wrapper
```

💡 Type `menu` to go back.
"""})
            elif user_input == '3':
                return jsonify({"response": """
⚙️ **Generators**  
Used to yield items one by one.

```python
def my_gen():  
    yield 1  
    yield 2

for x in my_gen():  
    print(x)
```

💡 Type `menu` to go back.
"""})

        if session.get('state') == 'ai_topics':
            if user_input == '1':
                return jsonify({"response": """
🤖 **What is AI?**  
AI allows machines to think and make decisions like humans.

💡 Type `menu` to go back.
"""})
            elif user_input == '2':
                return jsonify({"response": """
📊 **What is Machine Learning?**  
ML allows machines to learn from data and improve over time.

💡 Type `menu` to go back.
"""})
            elif user_input == '3':
                return jsonify({"response": """
💬 **What is a Chatbot?**  
A chatbot mimics human conversation using AI.

💡 Type `menu` to go back.
"""})

        return jsonify({"response": "❌ Invalid input. Type `menu` to restart."})

    chats = Chat.query.filter_by(user_id=current_user.id).all()
    return render_template('chat.html', chats=chats)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('chat'))
        else:
            flash('Invalid username or password', 'danger')
    return render_template('login.html', form=form, title="Login")

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash('Username already exists!', 'warning')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registered successfully! Please login.')
        return redirect(url_for('login'))
    return render_template('register.html', form=form, title="Register")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
