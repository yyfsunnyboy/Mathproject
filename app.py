# ==============================================================================
# 1. Imports
# ==============================================================================
import random
import os
import base64
import io
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import google.generativeai as genai
from PIL import Image
import time
from google.api_core.exceptions import ResourceExhausted

# ==============================================================================
# 2. App Initialization and Configuration
# ==============================================================================
app = Flask(__name__)
app.secret_key = 'GEMINI_API_KEY=AIzaSyAHdn-IImFJwyVMqRt5TdqBFOdnw_bgbbY'  # ★ 請務必更換成您自己的密鑰 ★
#app.secret_key = 'AIzaSyCCvlrh5-3Y_Ck15cZDJ-R0C3yYN9WTBpw' # ★ 備用的密鑰 ★

# --- Database Configuration ---
# 取得 instance 資料夾的絕對路徑
instance_path = app.instance_path
print(f"資料庫將會被儲存在: {instance_path}") # 加上這行方便您確認

# 確保 instance 資料夾存在
try:
    os.makedirs(instance_path)
except OSError:
    pass # 資料夾已存在

# 明確指定資料庫的完整路徑
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(instance_path, 'kumon_math.db')
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
# --- Gemini API Configuration ---
try:
    genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
    model = genai.GenerativeModel('models/gemini-pro-latest')
except Exception as e:
    print(f"Gemini API 尚未設定或金鑰錯誤: {e}")
    model = None

# ==============================================================================
# 3. Database Models
# ==============================================================================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    progress = db.relationship('UserProgress', backref='user', lazy=True)

class Skill(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    display_name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255))
    
    # vvvv 確保您已經加上這兩行 vvvv
    school_type = db.Column(db.String(20), nullable=True, default='共同')
    grade_level = db.Column(db.String(20), nullable=True, default='國中')
    # ^^^^ 確保您已經加上這兩行 ^^^^

    # ... (Skill 模型裡的其他欄位) ...

class UserProgress(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    skill_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    consecutive_correct = db.Column(db.Integer, default=0)
    total_correct = db.Column(db.Integer, default=0)
    total_attempted = db.Column(db.Integer, default=0)
    consecutive_incorrect = db.Column(db.Integer, default=0)
    __table_args__ = (db.UniqueConstraint('user_id', 'skill_id', name='_user_skill_uc'),)

class SkillDependency(db.Model):
    """ 用來儲存技能依賴關係 (知識圖譜) 的模型 """
    id = db.Column(db.Integer, primary_key=True)

    # '先備知識' 的 ID
    prerequisite_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)
    # '目標技能' 的 ID
    target_id = db.Column(db.Integer, db.ForeignKey('skill.id'), nullable=False)

    # 建立關係，讓我們可以方便地查詢
    prerequisite = db.relationship('Skill', foreign_keys=[prerequisite_id], backref='leading_to')
    target = db.relationship('Skill', foreign_keys=[target_id], backref='requires')

    # 確保同一個依賴關係不會重複
    db.UniqueConstraint('prerequisite_id', 'target_id', name='unique_dependency')

    def __repr__(self):
        return f'<Dependency: {self.prerequisite.display_name} -> {self.target.display_name}>'

# ^^^^ 程式碼加到這裡為止 ^^^^

# ==============================================================================
# 4. Helper Functions (Formatting, Checking)
# ==============================================================================

# --- Validation Functions (Referenced by name) ---
def validate_remainder(user_answer, correct_answer):
    # 簡單的字串比較
    return str(user_answer).strip().lower() == str(correct_answer).strip().lower()

def validate_factor(user_answer, correct_answer):
    # 判斷 '是' 或 '否'
    return str(user_answer).strip() == str(correct_answer)

def validate_linear_equation(user_answer, correct_answer):
    # 判斷 x=... 或 y=...
    return str(user_answer).strip().lower() == str(correct_answer).strip().lower()

def validate_check_point(user_answer, correct_answer):
    # 判斷 '是' 或 '否'
    return str(user_answer).strip() == str(correct_answer)

# --- Formatting Functions ---
def format_polynomial(coeffs):
    """將係數列表轉換成漂亮的多項式字串"""
    terms = []
    degree = len(coeffs) - 1
    for i, coeff in enumerate(coeffs):
        power = degree - i
        if coeff == 0:
            continue
        term = ""
        if coeff > 0:
            if i > 0:
                term += " + "
        else:
            term += " - "
        abs_coeff = abs(coeff)
        if abs_coeff != 1 or power == 0:
            term += str(abs_coeff)
        if power == 1:
            term += "x"
        elif power > 1:
            term += f"x^{power}"
        terms.append(term)
    if not terms:
        return "0"
    return "".join(terms)

def format_linear_equation_lhs(a, b):
    """將係數 (a, b) 轉換成 "ax + by" 的漂亮字串"""
    terms = []
    if a == 1:
        terms.append("x")
    elif a == -1:
        terms.append("-x")
    elif a != 0:
        terms.append(f"{a}x")
    if b > 0:
        if a != 0:
            terms.append(" + ")
        if b == 1:
            terms.append("y")
        else:
            terms.append(f"{b}y")
    elif b < 0:
        if a != 0:
            terms.append(" - ")
        else:
            terms.append("-")
        if b == -1:
            terms.append("y")
        else:
            terms.append(f"{abs(b)}y")
    if not terms:
        return "0"
    return "".join(terms)

def check_inequality(a, b, c, sign, x, y):
    """檢查點 (x, y) 是否滿足 ax + by [sign] c"""
    lhs = (a * x) + (b * y)
    if sign == '>':
        return lhs > c
    if sign == '>=':
        return lhs >= c
    if sign == '<':
        return lhs < c
    if sign == '<=':
        return lhs <= c
    return False

def format_inequality(a, b, c, sign):
    """將係數 (a, b, c) 和符號轉換成 "ax + by [sign] c" 的字串"""
    lhs_str = format_linear_equation_lhs(a, b)
    return f"{lhs_str} {sign} {c}"

# ==============================================================================
# 5. Question Generators
# ==============================================================================
def generate_remainder_theorem_question():
    """動態生成一道「餘式定理」的題目 (二次式或三次式)"""
    degree = random.choice([2, 3])
    k = random.randint(-3, 3)
    coeffs = []
    correct_answer = 0
    if degree == 2:
        a = random.randint(-3, 3)
        while a == 0:
            a = random.randint(-3, 3)
        b = random.randint(-5, 5)
        c = random.randint(-9, 9)
        coeffs = [a, b, c]
        correct_answer = (a * (k**2)) + (b * k) + c
    elif degree == 3:
        a = random.randint(-2, 2)
        while a == 0:
            a = random.randint(-2, 2)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        d = random.randint(-9, 9)
        coeffs = [a, b, c, d]
        correct_answer = (a * (k**3)) + (b * (k**2)) + (c * k) + d
    poly_text = format_polynomial(coeffs)
    k_sign = "-" if k >= 0 else "+"
    k_abs = abs(k)
    divisor_text = "(x)" if k == 0 else f"(x {k_sign} {k_abs})"
    question_text = f"求 f(x) = {poly_text} 除以 {divisor_text} 的餘式。"
    return {
        "text": question_text,
        "answer": str(correct_answer),
        "validation_function_name": validate_remainder.__name__
    }

def generate_factor_theorem_question():
    """動態生成一道「因式定理」的題目 (是/否)"""
    degree = random.choice([2, 3])
    k = random.randint(-3, 3)
    coeffs = []
    is_factor = random.choice([True, False])
    if degree == 2:
        a = random.randint(-3, 3)
        while a == 0:
            a = random.randint(-3, 3)
        b = random.randint(-5, 5)
        if is_factor:
            c = -((a * (k**2)) + (b * k))
        else:
            c = random.randint(-9, 9)
            remainder = (a * (k**2)) + (b * k) + c
            while remainder == 0:
                c = random.randint(-9, 9)
                remainder = (a * (k**2)) + (b * k) + c
        coeffs = [a, b, c]
    elif degree == 3:
        a = random.randint(-2, 2)
        while a == 0:
            a = random.randint(-2, 2)
        b = random.randint(-3, 3)
        c = random.randint(-5, 5)
        if is_factor:
            d = -((a * (k**3)) + (b * (k**2)) + (c * k))
        else:
            d = random.randint(-9, 9)
            remainder = (a * (k**3)) + (b * (k**2)) + (c * k) + d
            while remainder == 0:
                d = random.randint(-9, 9)
                remainder = (a * (k**3)) + (b * (k**2)) + (c * k) + d
        coeffs = [a, b, c, d]
    poly_text = format_polynomial(coeffs)
    k_sign = "-" if k >= 0 else "+"
    k_abs = abs(k)
    divisor_text = "(x)" if k == 0 else f"(x {k_sign} {k_abs})"
    question_text = f"請問 {divisor_text} 是否為 f(x) = {poly_text} 的因式？ (請回答 '是' 或 '否')"
    correct_answer = "是" if is_factor else "否"
    return {
        "text": question_text,
        "answer": correct_answer,
        "validation_function_name": validate_factor.__name__
    }

def generate_substitution_question():
    """動態生成一道「帶入消去法」的題目 (確保唯一解)。"""
    x_sol = random.randint(-5, 5)
    y_sol = random.randint(-5, 5)
    while x_sol == 0 or y_sol == 0:
        x_sol = random.randint(-5, 5)
        y_sol = random.randint(-5, 5)
    if random.choice([True, False]):  # 產生 y = mx + k
        m = random.randint(-3, 3)
        while m == 0:
            m = random.randint(-3, 3)
        k = y_sol - (m * x_sol)
        eq1_lhs = "y"
        eq1_rhs = f"{m}x"
        if k > 0:
            eq1_rhs += f" + {k}"
        elif k < 0:
            eq1_rhs += f" - {abs(k)}"
        a = random.randint(-3, 3)
        b = random.randint(-3, 3)
        while a == 0 or b == 0 or a == -m * b:
            a = random.randint(-3, 3)
            b = random.randint(-3, 3)
        c = (a * x_sol) + (b * y_sol)
        eq2_lhs = format_linear_equation_lhs(a, b)
        eq2_rhs = str(c)
    else:  # 產生 x = my + k
        m = random.randint(-3, 3)
        while m == 0:
            m = random.randint(-3, 3)
        k = x_sol - (m * y_sol)
        eq1_lhs = "x"
        eq1_rhs = f"{m}y"
        if k > 0:
            eq1_rhs += f" + {k}"
        elif k < 0:
            eq1_rhs += f" - {abs(k)}"
        a = random.randint(-3, 3)
        b = random.randint(-3, 3)
        while a == 0 or b == 0 or b == -m * a:
            a = random.randint(-3, 3)
            b = random.randint(-3, 3)
        c = (a * x_sol) + (b * y_sol)
        eq2_lhs = format_linear_equation_lhs(a, b)
        eq2_rhs = str(c)
    ask_for = random.choice(["x", "y"])
    answer = str(x_sol) if ask_for == "x" else str(y_sol)
    question_text = (f"請用帶入消去法解下列聯立方程式：\n"
                    f"  {eq1_lhs:<15} = {eq1_rhs:<10} ...... (1)\n"
                    f"  {eq2_lhs:<15} = {eq2_rhs:<10} ...... (2)\n\n"
                    f"請問 {ask_for} = ?")
    return {
        "text": question_text,
        "answer": answer,
        "validation_function_name": validate_linear_equation.__name__
    }

def generate_addition_subtraction_question():
    """動態生成一道「加減消去法」的題目 (加入倍數變化)。"""
    x_sol = random.randint(-5, 5)
    y_sol = random.randint(-5, 5)
    while x_sol == 0 or y_sol == 0:
        x_sol = random.randint(-5, 5)
        y_sol = random.randint(-5, 5)
    a1 = random.randint(-5, 5)
    b1 = random.randint(-5, 5)
    while a1 == 0 or b1 == 0:
        a1 = random.randint(-5, 5)
        b1 = random.randint(-5, 5)
    multiplier = random.choice([-3, -2, 2, 3])
    b2 = b1 * multiplier
    a2 = random.randint(-5, 5)
    while a2 == 0 or a2 == a1 * multiplier:
        a2 = random.randint(-5, 5)
    c1 = (a1 * x_sol) + (b1 * y_sol)
    c2 = (a2 * x_sol) + (b2 * y_sol)
    eq1_lhs = format_linear_equation_lhs(a1, b1)
    eq2_lhs = format_linear_equation_lhs(a2, b2)
    ask_for = random.choice(["x", "y"])
    answer = str(x_sol) if ask_for == "x" else str(y_sol)
    question_text = (f"請用加減消去法解下列聯立方程式：\n"
                    f"  {eq1_lhs:<15} = {c1:<10} ...... (1)\n"
                    f"  {eq2_lhs:<15} = {c2:<10} ...... (2)\n\n"
                    f"請問 {ask_for} = ?")
    return {
        "text": question_text,
        "answer": answer,
        "validation_function_name": validate_linear_equation.__name__
    }

def generate_check_point_in_system_question():
    """動態生成一道「判斷點是否為不等式系統解」的題目。"""
    num_inequalities = random.choice([2, 3])
    inequalities = []
    inequality_strs = []
    for _ in range(num_inequalities):
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
        while a == 0 and b == 0:
            a = random.randint(-5, 5)
            b = random.randint(-5, 5)
        temp_x = random.randint(-3, 3)
        temp_y = random.randint(-3, 3)
        c = (a * temp_x) + (b * temp_y)
        sign = random.choice(['>', '>=', '<', '<='])
        inequalities.append({'a': a, 'b': b, 'c': c, 'sign': sign})
        inequality_strs.append(format_inequality(a, b, c, sign))
    test_x = random.randint(-5, 5)
    test_y = random.randint(-5, 5)
    is_solution = True
    for ieq in inequalities:
        if not check_inequality(ieq['a'], ieq['b'], ieq['c'], ieq['sign'], test_x, test_y):
            is_solution = False
            break
    correct_answer = "是" if is_solution else "否"
    system_str = "\n".join([f"  {s}" for s in inequality_strs])
    question_text = f"請問點 ({test_x}, {test_y}) 是否為下列不等式系統的解？ (請回答 '是' 或 '否')\n{system_str}"
    return {
        "text": question_text,
        "answer": correct_answer,
        "validation_function_name": validate_check_point.__name__
    }

def generate_inequality_region_question():
    """動態生成一道「圖示不等式解區域」的題目。"""
    a = random.randint(-5, 5)
    b = random.randint(-5, 5)
    while a == 0 and b == 0:
        a = random.randint(-5, 5)
        b = random.randint(-5, 5)
    c = random.randint(-9, 9)
    while c == 0:
        c = random.randint(-9, 9)
    sign = random.choice(['>', '<', '>=', '<='])
    inequality_lhs = format_linear_equation_lhs(a, b)
    c_str = ""
    if c > 0:
        c_str = f" + {c}"
    elif c < 0:
        c_str = f" - {abs(c)}"
    inequality_expression = f"{inequality_lhs}{c_str}"
    full_inequality_string = f"{inequality_expression} {sign} 0"
    question_text = (
        f"請在下方的「數位計算紙」上，圖示二元一次不等式：\n\n"
        f"    {full_inequality_string}\n\n"
        f"畫完後，請點擊「AI 檢查計算」按鈕。"
    )
    return {
        "text": question_text,
        "answer": None,
        "validation_function_name": None,
        "inequality_string": full_inequality_string
    }

# ==============================================================================
# 6. Skill Engine Definition
# ==============================================================================
SKILL_ENGINE = {
    'remainder-theorem': {
        'generator': generate_remainder_theorem_question,
        'display_name': '餘式定理',
        'description': '練習 f(x) 除以 (x-k) 的餘式。',
        'prerequisite_skill_id': None
    },
    'factor-theorem': {
        'generator': generate_factor_theorem_question,
        'display_name': '因式定理',
        'description': '判斷 (x-k) 是否為 f(x) 的因式。',
        'prerequisite_skill_id': 'remainder-theorem'
    },
    'linear-eq-substitution': {
        'generator': generate_substitution_question,
        'display_name': '二元一次 (帶入消去法)',
        'description': '練習 y=ax+b 形式的帶入消去。',
        'prerequisite_skill_id': None
    },
    'linear-eq-addition': {
        'generator': generate_addition_subtraction_question,
        'display_name': '二元一次 (加減消去法)',
        'description': '練習係數需乘以倍數的加減消去。',
        'prerequisite_skill_id': 'linear-eq-substitution'
    },
    'linear-ineq-region': {
        'generator': generate_inequality_region_question,
        'display_name': '二元一次不等式 (圖解區域)',
        'description': '在數位計算紙上畫出不等式的解區域。',
        'prerequisite_skill_id': 'linear-eq-addition'
    },
    'linear-ineq-check-point': {
        'generator': generate_check_point_in_system_question,
        'display_name': '二元一次不等式 (判斷解)',
        'description': '判斷一個點是否為不等式系統的解。',
        'prerequisite_skill_id': 'linear-ineq-region'
    }
}

DEMOTION_THRESHOLD = 3  # 連續答錯 3 題就降級

def initialize_skills():
    """同步 SKILL_ENGINE 到資料庫 (包含先備知識)"""
    print("正在同步技能到資料庫...")
    for skill_id, skill_data_in_code in SKILL_ENGINE.items():
        skill_in_db = Skill.query.filter_by(skill_id=skill_id).first()
        needs_update = False
        if skill_in_db:
            # 比較並更新現有技能
            if skill_in_db.display_name != skill_data_in_code['display_name']:
                skill_in_db.display_name = skill_data_in_code['display_name']
                needs_update = True
            if 'description' in skill_data_in_code and skill_in_db.description != skill_data_in_code['description']:
                skill_in_db.description = skill_data_in_code['description']
                needs_update = True
            if skill_in_db.prerequisite_skill_id != skill_data_in_code.get('prerequisite_skill_id'):
                skill_in_db.prerequisite_skill_id = skill_data_in_code.get('prerequisite_skill_id')
                needs_update = True
            if needs_update:
                db.session.commit()
                print(f"更新技能 {skill_id} 到資料庫")
        else:
            # 創建新技能
            new_skill = Skill(
                skill_id=skill_id,
                display_name=skill_data_in_code['display_name'],
                description=skill_data_in_code.get('description', '無描述'),
                prerequisite_skill_id=skill_data_in_code.get('prerequisite_skill_id')
            )
            db.session.add(new_skill)
            db.session.commit()
            print(f"添加新技能 {skill_id} 到資料庫")

# ==============================================================================
# 7. Routes (View Functions)
# ==============================================================================

# --- Authentication Routes ---
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("帳號或密碼不可為空", "warning")
            return redirect(url_for('register'))
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash("這個帳號名稱已經有人用了！", "warning")
            return redirect(url_for('register'))
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("註冊成功！請登入。", "success")
        return redirect(url_for('login'))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash("帳號或密碼不可為空", "danger")
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and bcrypt.check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash(f"歡迎回來，{user.username}！", "success")
            return redirect(url_for('home'))
        else:
            flash("帳號或密碼錯誤。", "danger")
            return redirect(url_for('login'))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("您已成功登出。", "info")
    return redirect(url_for('login'))

# --- Core Application Routes ---
@app.route("/")
def home():
    if 'user_id' not in session:
        flash("請先登入！", "warning")
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route("/dashboard")
def dashboard():
    if 'user_id' not in session:
        flash("請先登入！", "warning")
        return redirect(url_for('login'))
    user_id = session['user_id']
    all_skills = Skill.query.all()
    user_progresses = UserProgress.query.filter_by(user_id=user_id).all()
    progress_map = {p.skill_id: p for p in user_progresses}
    dashboard_data = []
    for skill in all_skills:
        progress = progress_map.get(skill.id)
        dashboard_data.append({
            'skill': skill,
            'consecutive_correct': progress.consecutive_correct if progress else 0,
            'total_attempted': progress.total_attempted if progress else 0
        })
    return render_template('dashboard.html',
                           dashboard_data=dashboard_data,
                           username=session.get('username'))

@app.route("/practice/<string:skill_id>")
def practice(skill_id):
    if 'user_id' not in session:
        flash("請先登入！", "warning")
        return redirect(url_for('login'))
    skill = Skill.query.filter_by(skill_id=skill_id).first()
    if not skill or skill_id not in SKILL_ENGINE:
        flash("找不到指定的練習單元。", "danger")
        return redirect(url_for('dashboard'))
        
    question_data = SKILL_ENGINE[skill_id]['generator']()
    session['current_skill_id'] = skill_id
    session['current_question_text'] = question_data.get('text')
    session['current_answer'] = question_data.get('answer')
    session['current_inequality_string'] = question_data.get('inequality_string')
    session['validation_function_name'] = question_data.get('validation_function_name')
    
    print(f"({skill_id}) 新題目: {question_data.get('text')} (答案: {question_data.get('answer')})")
    
    return render_template('index.html',
                           question_text=question_data.get('text'),
                           inequality_string=question_data.get('inequality_string') or '',
                           username=session.get('username'),
                           skill_display_name=skill.display_name)

# --- API Endpoints ---
@app.route("/get_next_question", methods=["GET"])
def get_next_question():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    skill_id = session.get('current_skill_id')
    if not skill_id or skill_id not in SKILL_ENGINE:
        return jsonify({"error": "Skill error"}), 400
        
    generator_func = SKILL_ENGINE[skill_id]['generator']
    question_data = generator_func()
    
    session['current_answer'] = question_data.get('answer')
    session['current_question_text'] = question_data.get('text')
    session['current_inequality_string'] = question_data.get('inequality_string')
    session['validation_function_name'] = question_data.get('validation_function_name')
    
    print(f"({skill_id}) 下一題: {question_data.get('text')} (答案: {question_data.get('answer')})")

    return jsonify({
        "new_question_text": question_data.get('text'),
        "inequality_string": question_data.get('inequality_string')
    })

@app.route("/check_answer", methods=["POST"])
def check_answer():
    if 'user_id' not in session:
        return jsonify({"error": "Not logged in"}), 401
    data = request.get_json()
    if not data or 'answer' not in data:
        return jsonify({"error": "Missing JSON data or 'answer'"}), 400
        
    user_answer = data.get('answer')
    skill_id_str = session.get('current_skill_id')
    correct_answer = session.get('current_answer')
    validation_func_name = session.get('validation_function_name')
    if not skill_id_str:
        return jsonify({"error": "Session missing skill_id"}), 400
    
    is_correct = False
    result_message = ""
    validation_func = globals().get(validation_func_name) if validation_func_name else None
    
    if validation_func:
        try:
            is_correct = validation_func(user_answer, correct_answer)
            result_message = "答對了！" if is_correct else f"答錯了... (提示: {correct_answer})"
        except Exception as e:
            print(f"Validation function error: {e}")
            is_correct = False
            result_message = "答案格式錯誤"
    else:
        is_correct = (str(user_answer).strip().lower() == str(correct_answer).strip().lower())
        result_message = "答對了！" if is_correct else f"答錯了... (提示: {correct_answer})"
    
    demote_to_skill_id = None
    
    try:
        user_id = session['user_id']
        skill = Skill.query.filter_by(skill_id=skill_id_str).first()
        if skill:
            progress = UserProgress.query.filter_by(user_id=user_id, skill_id=skill.id).first()
            if not progress:
                progress = UserProgress(user_id=user_id, skill_id=skill.id)
                db.session.add(progress)
            
            progress.total_attempted += 1
            if is_correct:
                progress.consecutive_correct += 1
                progress.total_correct += 1
                progress.consecutive_incorrect = 0
            else:
                progress.consecutive_correct = 0
                progress.consecutive_incorrect += 1
                if progress.consecutive_incorrect >= DEMOTION_THRESHOLD and skill.prerequisite_skill_id:
                    demote_to_skill_id = skill.prerequisite_skill_id
                    prereq_skill = Skill.query.filter_by(skill_id=demote_to_skill_id).first()
                    prereq_name = prereq_skill.display_name if prereq_skill else "基礎單元"
                    result_message = f"您在「{skill.display_name}」單元連續答錯 {progress.consecutive_incorrect} 題了。\n系統建議您先回去複習「{prereq_name}」！"
                    progress.consecutive_incorrect = 0
            db.session.commit()
        else:
            print(f"Warning: Skill '{skill_id_str}' not found.")
    except Exception as e:
        db.session.rollback()
        print(f"Error updating progress: {e}")
    
    return jsonify({
        "result": result_message,
        "correct": is_correct,
        "demote_to_skill_id": demote_to_skill_id
    })

@app.route("/ask_gemini", methods=["POST"])
def ask_gemini():
    if 'user_id' not in session:
        return jsonify({"reply": "Not logged in"}), 401
    if model is None:
        return jsonify({"reply": "AI 助教尚未設定。"}), 500
    data = request.get_json()
    if not data or not data.get('prompt') or not data.get('current_question'):
        return jsonify({"reply": "錯誤：缺少提示或題目內容。"}), 400
         
    user_prompt = data.get('prompt')
    current_question = data.get('current_question')
    current_skill_id = session.get('current_skill_id', 'unknown')
    current_skill_display_name = SKILL_ENGINE.get(current_skill_id, {}).get('display_name', '數學')
    
    system_instruction = f"""
        你是一位專業且有耐心的高中數學家教，專門輔導資源班的學生。
        學生的目標是段考及格。請用繁體中文回答。
        
        你的任務：
        1.  **角色扮演**：你是一位友善的 AI 助教。
        2.  **教學重點**：學生目前正在練習「{current_skill_display_name}」。
        3.  **當前題目**：學生正在看的題目是「{current_question}」。
        4.  **回答限制**：
            * **不要直接給答案！** 這是最重要的規則。
            * 如果學生問「這題答案是什麼？」，你應該反問他：「你覺得第一步該怎麼做呢？」或「你記得{current_skill_display_name}的定義嗎？」。
            * 如果學生問「詳解」，請提供「解題步驟」和「思路引導」，而不是只給計算過程。
            * 如果學生問觀念（例如「什麼是{current_skill_display_name}？」），請用最簡單、最白話的方式解釋。
    
        學生的問題是：「{user_prompt}」
        請根據上述規則，提供你的回答：
        """
    try:
        response = model.generate_content(system_instruction)
        ai_reply = response.text
    except Exception as e:
        print(f"Gemini API 呼叫失敗: {e}")
        ai_reply = "抱歉，助教現在有點忙... 請稍後再試。"
    return jsonify({"reply": ai_reply})

@app.route("/analyze_handwriting", methods=["POST"])
def analyze_handwriting():
    if 'user_id' not in session:
        return jsonify({"reply": "Not logged in"}), 401
    if model is None:
        return jsonify({"reply": "AI 助教尚未設定。"}), 500

    # --- 獲取 Session 中的情境 ---
    user_id = session.get('user_id')
    current_skill_id_str = session.get('current_skill_id', 'unknown')
    current_question = session.get('current_question_text', '未知題目')
    current_answer = session.get('current_answer')  # 可能是 None
    current_inequality_string = session.get('current_inequality_string')  # 可能是 None
    current_skill_display_name = SKILL_ENGINE.get(current_skill_id_str, {}).get('display_name', '數學')
    
    # --- 獲取前端傳來的資料 ---
    data = request.get_json()
    if not data:
        return jsonify({"reply": "錯誤：未收到 JSON 資料。"}), 400
    image_data_url = data.get('image_data_url')
    
    if not image_data_url:
        print("錯誤: 前端未發送 image_data_url")
        return jsonify({"reply": "錯誤：缺少圖片資料。"}), 400

    try:
        # 2. 轉換圖片
        header, encoded = image_data_url.split(",", 1)
        image_data = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_data))

        # 3. 根據 current_inequality_string 是否存在，決定提示詞
        prompt_parts = []
        is_graph_question = bool(current_inequality_string)

        if is_graph_question:
            # --- 提示詞：畫圖題 (二元一次不等式) ---
            print(f"收到畫圖題分析請求: {current_inequality_string}")
            prompt_parts = [
                f"""你是一位專業且有耐心的數學家教，專門輔導高中資源班學生，目標是讓學生段考及格。
                請用繁體中文回答。學生正在練習「{current_skill_display_name}」單元，題目是：
                「{current_question}」
                學生提交了一張手繪圖片（已提供），請根據以下要求分析：
                - 題目要求學生在數位計算紙上畫出二元一次不等式 {current_inequality_string} 的解區域。
                - 檢查學生繪製的直線和陰影區域是否正確。
                - 如果正確，回覆格式為：「CORRECT: 畫得很好！解區域完全正確。」
                - 如果錯誤，回覆格式為：「INCORRECT: 錯誤的地方在...（具體說明，例如直線位置或陰影方向錯誤）」
                - 避免使用「請」或「請問」，直接給出結論。
                - 回覆中，如果是「INCORRECT」，請確保第一行是「INCORRECT: 錯誤的地方在...」，後續提供詳細解釋。
                以下是學生的手繪圖片：""",
                image
            ]
        else:
            # --- 提示詞：計算題 (通用) ---
            prompt_parts = [
                f"""你是一位專業且有耐心的數學家教，專門輔導高中資源班學生，目標是讓學生段考及格。
                請用繁體中文回答。學生正在練習「{current_skill_display_name}」單元，題目是：
                「{current_question}」
                學生提交了一張手寫計算過程的圖片（已提供），請根據以下要求分析：
                - 檢查計算過程是否正確。
                - 如果正確，回覆格式為：「CORRECT: 計算正確！」
                - 如果錯誤，回覆格式為：「INCORRECT: 錯誤的地方在...（具體說明，例如某一步計算錯誤）」
                - 避免使用「請」或「請問」，直接給出結論。
                以下是學生的手寫計算過程：""",
                image
            ]

        # 4. 呼叫 Gemini API
        response = model.generate_content(prompt_parts)
        ai_reply = response.text.strip()

        # 5. 解讀 AI 回覆並判斷對錯 (只對畫圖題更新進度)
        is_graph_correct = False
        demote_to_skill_id = None  # 初始化 demote_to_skill_id
        short_feedback = ai_reply.split('\n')[0] if ai_reply else "分析錯誤"
        detailed_feedback = ai_reply if ai_reply else "分析失敗，請重試。"

        if is_graph_question:
            if ai_reply.startswith("CORRECT:"):
                is_graph_correct = True
            elif ai_reply.startswith("INCORRECT:"):
                is_graph_correct = False
                # 確保 detailed_feedback 包含完整回覆
                detailed_feedback = ai_reply
            else:
                is_graph_correct = False
                short_feedback = f"AI 回覆格式錯誤...\n({ai_reply})"
                detailed_feedback = short_feedback

            # --- 更新資料庫進度 (只針對畫圖題) ---
            try:
                skill = Skill.query.filter_by(skill_id=current_skill_id_str).first()
                if skill and user_id:
                    progress = UserProgress.query.filter_by(user_id=user_id, skill_id=skill.id).first()
                    if not progress:
                        progress = UserProgress(user_id=user_id, skill_id=skill.id)
                        db.session.add(progress)
                    progress.total_attempted += 1
                    if is_graph_correct:
                        progress.consecutive_correct += 1
                        progress.total_correct += 1
                        progress.consecutive_incorrect = 0
                    else:
                        progress.consecutive_correct = 0
                        progress.consecutive_incorrect += 1
                        if progress.consecutive_incorrect >= DEMOTION_THRESHOLD and skill.prerequisite_skill_id:
                            demote_to_skill_id = skill.prerequisite_skill_id
                            prereq_skill = Skill.query.filter_by(skill_id=demote_to_skill_id).first()
                            prereq_name = prereq_skill.display_name if prereq_skill else "基礎單元"
                            detailed_feedback += f"\n\n錯誤次數較多，建議您先複習「{prereq_name}」。"
                    db.session.commit()
                    print(f"畫圖題進度已更新: correct={is_graph_correct}, demote={demote_to_skill_id}")
                else:
                    print("警告: 找不到技能或用戶，無法更新畫圖題進度")
            except Exception as e:
                db.session.rollback()
                print(f"Error updating progress: {e}")

    except Exception as e:
        print(f"Gemini API 或圖片處理失敗: {e}")
        is_graph_correct = False
        demote_to_skill_id = None  # 確保在異常情況下也有預設值
        short_feedback = f"分析失敗：{str(e)[:100]}... 請檢查圖片或稍後再試。"
        detailed_feedback = short_feedback

    # 7. 回傳結果給前端
    print(f"回傳給前端: short_feedback='{short_feedback}', detailed_feedback='{detailed_feedback}', is_graph_correct={is_graph_correct}, demote={demote_to_skill_id}")
    return jsonify({
        "short_feedback": short_feedback,  # 左邊紅色區塊顯示
        "reply": detailed_feedback,        # 右邊對話框顯示
        "is_graph_correct": is_graph_correct,
        "demote_to_skill_id": demote_to_skill_id
    })

# ==============================================================================
# 8. Application Runner
# ==============================================================================
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # 確保所有資料表都建立
        initialize_skills()  # 同步技能列表
    print("Starting Flask app...")
    app.run(debug=True)