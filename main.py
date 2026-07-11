from flask import Flask, render_template, request
from ultralytics import YOLO
import os, uuid, cv2, numpy as np
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SQLALCHEMY_DATABASE_URI"] = "mysql+pymysql://root:Theint12345%40@localhost/rice_growth_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

# Database model
class AnalysisRecord(db.Model):
    __tablename__ = "AnalysisRecord"
    id = db.Column(db.Integer, primary_key=True)
    variety = db.Column(db.String(50), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    height = db.Column(db.Float, nullable=False)
    weed_count = db.Column(db.Integer, nullable=False)
    growth_status = db.Column(db.String(50), nullable=False)
    advice = db.Column(db.String(200), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

# YOLO model
model = YOLO("models/best (2).pt")

# Growth thresholds
growth_thresholds = {
    "Paw San Hmwe": {7:[4,6],14:[12,15],30:[27,30]},
    "Shwebo Paw San": {7:[4.5,6],14:[13,15],30:[30,34]},
    "Manawthukha": {7:[4.5,6],14:[13,16],30:[32,36]},
    "Sin Thwe Latt": {7:[4,5],14:[12,14],30:[28,31]},
    "Shwe Thwe Yin": {7:[4,5],14:[13,15],30:[29,32]},
    "Eyar Min": {7:[4,5],14:[12,14],30:[28,30]},
    "Ngasein": {7:[4,5],14:[13,15],30:[29,32]},
    "Yadanarbon": {7:[4.5,6],14:[13,15],30:[30,32]},
    "Ayeyar Min": {7:[4,5],14:[12,14],30:[28,31]},
    "Hsin Yin Aung": {7:[4.5,6],14:[13,15],30:[30,33]}
}

# Height measurement
def measure_height(image_path, ruler_cm=30, marker_cm=None, save_path=None):
    img = cv2.imread(image_path)
    if img is None: return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    cm_per_pixel = None
    if lines is not None:
        y_coords = []
        for l in lines:
            x1,y1,x2,y2 = l[0]
            if abs(x1-x2) < 5:
                y_coords.extend([y1,y2])
        if y_coords:
            ruler_top, ruler_bottom = min(y_coords), max(y_coords)
            cm_per_pixel = ruler_cm / (ruler_bottom - ruler_top)

    if cm_per_pixel is None: return None

    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    rice = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(rice)

    height_cm = h * cm_per_pixel
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 4)
    cv2.line(img, (x,y+h), (x,y), (255,255,0), 4)
    cv2.putText(img, f"Rice Height: {round(height_cm,2)} cm",
            (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,0), 3)
    if save_path:
        cv2.imwrite(save_path, img)

    return height_cm

# Growth stage
def growth_check(age):
    if age < 20: return {"en":"Seedling","mm":"အပင်ငယ်အဆင့်"}
    elif age < 45: return {"en":"Tillering","mm":"အပင်ခွဲပေါက်အဆင့်"}
    elif age < 70: return {"en":"Vegetative","mm":"ပင်စည်ကြီးထွားအဆင့်"}
    else: return {"en":"Heading","mm":"နှံထွက်အဆင့်"}

# Growth threshold check
def check_growth(variety, age, height):
    if variety not in growth_thresholds:
        return True
    thresholds = growth_thresholds[variety]
    nearest_age = min(thresholds.keys(), key=lambda k: abs(k-age))
    low, high = thresholds[nearest_age]
    return low <= height <= high

# Weed detection
def detect_weed(path, save_path):
    result = model.predict(path, conf=0.25)[0]
    count = len(result.boxes)
    cv2.imwrite(save_path, result.plot())
    weed = {"en":"Weed Detected","mm":"ပေါင်းမြက်တွေ့ရှိသည်"} if count>0 else {"en":"No Weed","mm":"ပေါင်းမြက်မတွေ့ပါ"}
    return weed, count

@app.route("/")
def home():
    return render_template("index.html",
        growth_en="Waiting...", growth_mm="စောင့်ဆိုင်းနေသည်...",
        weed_en="Waiting...", weed_mm="စောင့်ဆိုင်းနေသည်...",
        count=0, height=0, height_status_en="Waiting...", height_status_mm="စောင့်ဆိုင်းနေသည်...",
        advice_en="-", advice_mm="-",
        image_urls=[], yolo_urls=[]
    )

@app.route("/predict", methods=["POST"])
def predict():
    weed_img = request.files.get("weed_image")
    ruler_img = request.files.get("ruler_image")

    if not weed_img or not ruler_img:
        return "Please upload both Weed Detection and Height Measurement images"

    age = int(request.form.get("age", 0))
    variety = request.form.get("variety", "Paw San Hmwe")

    weed_filename = str(uuid.uuid4()) + "_" + secure_filename(weed_img.filename)
    ruler_filename = str(uuid.uuid4()) + "_" + secure_filename(ruler_img.filename)
    weed_path = os.path.join(UPLOAD_FOLDER, weed_filename)
    ruler_path = os.path.join(UPLOAD_FOLDER, ruler_filename)
    weed_img.save(weed_path)
    ruler_img.save(ruler_path)

    # Weed detection
    weed_result_path = os.path.join(RESULT_FOLDER, weed_filename)
    weed, count = detect_weed(weed_path, weed_result_path)

    # Height measurement
    height_result_path = os.path.join(RESULT_FOLDER, ruler_filename)
    height = measure_height(ruler_path, ruler_cm=30, save_path=height_result_path) or 0

    # Growth stage
    growth = growth_check(age)

    # Advice logic
    if not check_growth(variety, age, height):
        if count > 0:
            advice = {
                "en": f"Weed detected in {variety}. Apply herbicide + nutrient support.",
                "mm": f"{variety} တွင် ပေါင်းမြက်တွေ့ရှိသည်။ ပေါင်းသတ်ဆေးအသုံးပြုပြီး အာဟာရဖြည့်တင်းပေးပါ။"
            }
        else:
            advice = {
                "en": f"{variety} growth is slow. No weeds detected. Recheck in 7 days.",
                "mm": f"{variety} ကြီးထွားမှုနှေးနေပါသည်။ သို့သော် ပေါင်းမြက်မတွေ့ပါ။ (၇) ရက်အကြာတွင် ပြန်လည်စစ်ဆေးပါ။"
            }
    else:
        if count > 0:
            advice = {
                "en": f"{variety} Growth is normal, but weeds detected. Apply herbicide.",
                "mm": f"{variety} ကြီးထွားမှု ပုံမှန်ဖြစ်ပါသည်။ သို့သော် ပေါင်းမြက်တွေ့ရှိသည်။ ပေါင်းသတ်ဆေးအသုံးပြုပါ။"
            }
        else:
            advice = {
                "en": f"{variety} growth is normal. No weeds detected.",
                "mm": f"{variety} ကြီးထွားမှု ပုံမှန်ဖြစ်ပါသည်။ ပေါင်းမြက်မတွေ့ပါ။"
            }

    # Height status
    height_status_en = "Normal" if check_growth(variety, age, height) else "Slow"
    height_status_mm = "ပုံမှန်" if check_growth(variety, age, height) else "နှေး"
    # Save record
    record = AnalysisRecord(
        variety=variety,
        age=age,
        height=height,
        weed_count=count,
        growth_status=growth["mm"],
        advice=advice["mm"]
    )
    db.session.add(record)
    db.session.commit()

    return render_template("index.html",
    growth_en=growth["en"], growth_mm=growth["mm"],
    weed_en=weed["en"], weed_mm=weed["mm"],
    count=count, height=round(height, 2),
    height_status_en=height_status_en, height_status_mm=height_status_mm,
    advice_en=advice["en"], advice_mm=advice["mm"],
    image_urls=["/" + weed_path, "/" + ruler_path],
    yolo_urls=["/" + weed_result_path, "/" + height_result_path]
)

@app.route("/history")
def history():
    records = AnalysisRecord.query.order_by(AnalysisRecord.timestamp.asc()).all()
    return render_template("history.html", records=records)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
