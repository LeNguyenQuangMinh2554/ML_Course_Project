# 🏥 Dự án: Phân tích và Dự đoán Chi phí Nhập viện tại New York (2009)

> **Môn học:** Machine Learning - Kỳ thi cuối kỳ  
> **Đề tài:** Phân tích, dự đoán chi phí nhập viện tại các bệnh viện bang New York (chỉ năm 2009)

---

## 📚 MỤC LỤC

1. [Giới thiệu dự án](#-giới-thiệu-dự-án)
2. [Machine Learning là gì?](#-machine-learning-là-gì-giải-thích-cho-người-mới)
3. [Cấu trúc dự án](#-cấu-trúc-dự-án)
4. [Dataset - Bộ dữ liệu](#-dataset---bộ-dữ-liệu)
5. [Quy trình thực hiện (Pipeline)](#-quy-trình-thực-hiện-pipeline)
6. [Giải thích từng phần trong Notebook](#-giải-thích-từng-phần-trong-notebook)
7. [Ứng dụng GUI Demo](#-ứng-dụng-gui-demo)
8. [Cách chạy dự án](#-cách-chạy-dự-án)
9. [Kết quả và Kết luận](#-kết-quả-và-kết-luận)
10. [Thuật ngữ cần nhớ](#-thuật-ngữ-cần-nhớ)

---

## 🎯 Giới thiệu dự án

### Vấn đề cần giải quyết

Hãy tưởng tượng bạn là một bệnh nhân sắp nhập viện ở New York. Bạn muốn biết:

- **"Chi phí nhập viện của tôi sẽ là bao nhiêu?"**
- **"Những yếu tố nào ảnh hưởng đến chi phí?"**

Đây chính là vấn đề chúng ta giải quyết!

### Mục tiêu

1. **Phân tích** các yếu tố ảnh hưởng đến chi phí nhập viện
2. **Xây dựng mô hình** để dự đoán chi phí dựa trên thông tin bệnh nhân
3. **Tạo ứng dụng** cho phép nhập thông tin và nhận kết quả dự đoán

---

## 🤖 Machine Learning là gì? (Giải thích cho người mới)

### Định nghĩa đơn giản

**Machine Learning (Học máy)** = Dạy máy tính học từ dữ liệu để đưa ra dự đoán.

### Ví dụ thực tế

Hãy nghĩ về cách bạn học đoán giá nhà:

```
📖 Bạn xem 100 căn nhà đã bán:
   - Nhà 50m², 2 phòng ngủ, quận 1 → 5 tỷ
   - Nhà 100m², 3 phòng ngủ, quận 7 → 8 tỷ
   - Nhà 30m², 1 phòng ngủ, quận 9 → 2 tỷ
   - ... (97 căn nữa)

🧠 Não bạn TỰ ĐỘNG học ra quy luật:
   - Diện tích lớn → giá cao
   - Quận trung tâm → giá cao
   - Nhiều phòng ngủ → giá cao

❓ Khi thấy căn nhà mới (80m², 2PN, quận 3), bạn có thể ĐOÁN giá!
```

**Machine Learning làm ĐÚNG ĐIỀU NÀY**, nhưng thay vì não người, ta dùng máy tính!

### Trong dự án của chúng ta

```
📊 Dữ liệu: 1 triệu+ hồ sơ nhập viện đã có chi phí
   - Bệnh nhân A: 70 tuổi, nằm 5 ngày, cấp cứu → $50,000
   - Bệnh nhân B: 25 tuổi, nằm 2 ngày, tự nguyện → $15,000
   - ...

🤖 Máy tính HỌC quy luật:
   - Tuổi cao → chi phí cao
   - Nằm viện lâu → chi phí cao
   - Cấp cứu → chi phí cao hơn tự nguyện

❓ Bệnh nhân mới: 45 tuổi, nằm 3 ngày, cấp cứu → Dự đoán: $25,000
```

---

## 📁 Cấu trúc dự án

```
ML_Course_Project/
│
├── 📓 NY_Hospital_Charges_Analysis.ipynb   ← Notebook chính (phân tích + train model)
│
├── 🖥️ gui_app.py                           ← Ứng dụng web demo (Streamlit)
│
├── 📊 NY Hospital Admissions - Dataset.csv ← Dữ liệu gốc (1 triệu+ dòng)
│
├── 📋 criteria.md                          ← Tiêu chí chấm điểm của thầy
│
├── 📖 README.md                            ← File bạn đang đọc
│
└── 📁 .venv/                               ← Môi trường Python (tự động tạo)
```

---

## 📊 Dataset - Bộ dữ liệu

### Thông tin cơ bản

| Thông số              | Giá trị                   |
| --------------------- | ------------------------- |
| **Số dòng (records)** | 1,048,575 hồ sơ nhập viện |
| **Số cột (features)** | 11 đặc trưng              |
| **Năm**               | 2009                      |
| **Nguồn**             | Bệnh viện bang New York   |

### Giải thích các cột

| Tên cột          | Ý nghĩa              | Ví dụ                                          |
| ---------------- | -------------------- | ---------------------------------------------- |
| `Service Area`   | Khu vực dịch vụ y tế | New York City, Long Island, ...                |
| `County`         | Quận/Hạt             | Manhattan, Brooklyn, Queens, ...               |
| `Name`           | Tên bệnh viện        | NYU Hospital, Mount Sinai, ...                 |
| `Age Group`      | Nhóm tuổi            | 0-17, 18-29, 30-49, 50-69, 70+                 |
| `Gender`         | Giới tính            | M (Nam), F (Nữ)                                |
| `Race`           | Chủng tộc            | White, Black, Asian, ...                       |
| `Ethnicity`      | Dân tộc              | Hispanic, Non-Hispanic, ...                    |
| `Length of Stay` | Số ngày nằm viện     | 1, 2, 3, ..., 120 ngày                         |
| `Admission Type` | Loại nhập viện       | Emergency (Cấp cứu), Elective (Tự nguyện), ... |
| `Discharge Year` | Năm xuất viện        | 2009                                           |
| `Total Charges`  | **TỔNG CHI PHÍ** 💰  | $1 đến $4,000,000+                             |

### ⭐ Biến mục tiêu (Target Variable)

**`Total Charges`** = Cột chúng ta muốn DỰ ĐOÁN!

Các cột còn lại là **Features (Đặc trưng)** = Thông tin đầu vào để dự đoán.

```
          ┌─────────────────────────────────────────┐
INPUT     │  Age, Gender, Length of Stay, ...      │
(Features)│  "70 tuổi, Nam, nằm 5 ngày, cấp cứu"   │
          └───────────────────┬─────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   MÔ HÌNH ML    │
                    │  (đã học từ     │
                    │   1 triệu dữ    │
                    │   liệu cũ)      │
                    └────────┬────────┘
                             │
                             ▼
          ┌─────────────────────────────────────────┐
OUTPUT    │         Total Charges = $45,000        │
(Target)  │         "Chi phí dự đoán"              │
          └─────────────────────────────────────────┘
```

---

## 🔄 Quy trình thực hiện (Pipeline)

### Tổng quan 8 bước (theo yêu cầu của thầy)

```
┌──────────────────────────────────────────────────────────────────────┐
│                        QUY TRÌNH MACHINE LEARNING                    │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ① IMPORT          Nhập các thư viện cần thiết                      │
│       ↓                                                              │
│  ② DATASET         Load và xem tổng quan dữ liệu                    │
│    OVERVIEW                                                          │
│       ↓                                                              │
│  ③ DATA            Ghép nối dữ liệu (nếu có nhiều nguồn)            │
│    INTEGRATION                                                       │
│       ↓                                                              │
│  ④ DATA            Làm sạch: xóa trùng lặp, xử lý missing,         │
│    CLEANING        loại bỏ outliers                                  │
│       ↓                                                              │
│  ⑤ EDA             Khám phá dữ liệu bằng biểu đồ và thống kê       │
│    (Exploratory                                                      │
│     Data Analysis)                                                   │
│       ↓                                                              │
│  ⑥ FEATURE         Tạo biến mới từ dữ liệu có sẵn                  │
│    ENGINEERING                                                       │
│       ↓                                                              │
│  ⑦ EXPERIMENT      Thử nghiệm nhiều mô hình khác nhau               │
│       ↓                                                              │
│  ⑧ RESULT          So sánh và chọn mô hình tốt nhất                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 📓 Giải thích từng phần trong Notebook

### 1️⃣ Package Import - Nhập thư viện

```python
import pandas as pd        # Xử lý dữ liệu dạng bảng
import numpy as np         # Tính toán số học
import matplotlib.pyplot   # Vẽ biểu đồ
import seaborn as sns      # Vẽ biểu đồ đẹp hơn
from sklearn...            # Thư viện Machine Learning
```

**Giải thích đơn giản:**

- `pandas` = Excel trong Python (đọc file, lọc dữ liệu, tính toán)
- `numpy` = Máy tính khoa học (tính toán nhanh)
- `matplotlib/seaborn` = Vẽ biểu đồ
- `sklearn` = "Hộp công cụ ML" chứa sẵn các thuật toán

---

### 2️⃣ Dataset Overview - Tổng quan dữ liệu

**Mục đích:** Hiểu dữ liệu trước khi làm việc

```python
df = pd.read_csv('NY Hospital Admissions - Dataset.csv')  # Đọc file
df.shape        # Kích thước: (1048575, 11) = 1 triệu dòng, 11 cột
df.head()       # Xem 5 dòng đầu
df.info()       # Thông tin các cột
df.describe()   # Thống kê cơ bản (min, max, mean, ...)
```

**Những gì chúng ta phát hiện:**

- Dữ liệu có **1,048,575 hồ sơ** và **11 cột**
- Có **224 bệnh viện** khác nhau
- Chi phí trung bình: **~$29,617**
- Chi phí cao nhất: **>$4 triệu** (outlier!)
- Một số cột có **missing values** (dữ liệu bị thiếu)

---

### 3️⃣ Data Integration - Tích hợp dữ liệu

**Mục đích:** Ghép nhiều nguồn dữ liệu thành một

Trong dự án này, chúng ta chỉ có **1 file CSV**, nên bước này đơn giản.

Nếu có nhiều file (ví dụ: file bệnh nhân + file bệnh viện + file bảo hiểm), ta sẽ ghép chúng lại bằng `pd.merge()`.

---

### 4️⃣ Data Cleaning - Làm sạch dữ liệu

**Đây là bước RẤT QUAN TRỌNG!** Dữ liệu bẩn → Kết quả sai!

#### 4.1 Xử lý Duplicates (Dữ liệu trùng lặp)

```python
# Kiểm tra có dòng nào bị lặp không
df.duplicated().sum()  # → 0 (không có trùng lặp)

# Nếu có, xóa đi
df = df.drop_duplicates()
```

**Tại sao quan trọng?** Nếu cùng một bệnh nhân xuất hiện 10 lần, mô hình sẽ bị "thiên vị" về người đó!

#### 4.2 Xử lý Missing Data (Dữ liệu bị thiếu)

```python
df.isnull().sum()  # Đếm số ô trống mỗi cột
```

**Kết quả:**

- `Service Area`: 2,051 thiếu
- `County`: 2,051 thiếu
- `Length of Stay`: 791 thiếu

**Cách xử lý:**

```python
df = df.dropna()  # Xóa các dòng có dữ liệu thiếu
```

**Tại sao xóa?** Vì tỷ lệ thiếu rất nhỏ (~0.2%), xóa đi không ảnh hưởng nhiều.

#### 4.3 Xử lý Outliers (Dữ liệu ngoại lai)

**Outlier là gì?**

```
Ví dụ chi phí nhập viện:
- Đa số: $10,000 - $50,000
- Một vài người: $4,000,000 (!!!)

Người $4 triệu là OUTLIER - họ quá khác biệt so với đa số!
```

**Tại sao cần xử lý?**

- Outlier làm "nhiễu" mô hình
- Mô hình sẽ cố gắng fit cả những trường hợp cực đoan → dự đoán sai cho đa số

**Cách xử lý - Phương pháp IQR:**

```python
Q1 = df['Total Charges'].quantile(0.25)   # Phân vị 25%
Q3 = df['Total Charges'].quantile(0.75)   # Phân vị 75%
IQR = Q3 - Q1                              # Khoảng tứ phân vị

lower = Q1 - 1.5 * IQR   # Giới hạn dưới
upper = Q3 + 1.5 * IQR   # Giới hạn trên

# Giữ lại chỉ những giá trị trong khoảng hợp lệ
df = df[(df['Total Charges'] >= lower) & (df['Total Charges'] <= upper)]
```

**Minh họa:**

```
                    IQR
           ◄────────────────►
    ──────[────────|────────]──────
         Q1       Q2       Q3

    ◄────────────────────────────────►
    lower                         upper
    (Q1-1.5*IQR)            (Q3+1.5*IQR)

    ✓ Giữ lại: nằm trong [lower, upper]
    ✗ Loại bỏ: nằm ngoài khoảng này (outliers)
```

---

### 5️⃣ EDA - Exploratory Data Analysis (Khám phá dữ liệu)

**Mục đích:** Hiểu sâu về dữ liệu thông qua biểu đồ và thống kê

#### 5.1 Phân tích biến mục tiêu (Total Charges)

```python
# Histogram: xem phân phối chi phí
plt.hist(df['Total Charges'])
```

**Nhận xét:**

- Chi phí phân bố **lệch phải (right-skewed)**
- Đa số chi phí thấp, ít người chi phí cao
- Sau khi loại outliers, phân bố hợp lý hơn

#### 5.2 Phân tích các biến phân loại

```python
# Chi phí theo nhóm tuổi
df.groupby('Age Group')['Total Charges'].mean()
```

**Phát hiện thú vị:**

| Yếu tố             | Nhận xét                                   |
| ------------------ | ------------------------------------------ |
| **Tuổi**           | Người 70+ có chi phí cao nhất              |
| **Giới tính**      | Nam và Nữ tương đương                      |
| **Loại nhập viện** | Cấp cứu (Emergency) > Tự nguyện (Elective) |
| **Số ngày nằm**    | Càng lâu → càng tốn                        |
| **Bệnh viện**      | Chênh lệch lớn giữa các bệnh viện          |

#### 5.3 Visualization (Trực quan hóa)

Các loại biểu đồ sử dụng:

```
📊 HISTOGRAM (Biểu đồ tần suất)
   → Xem phân phối của 1 biến số

📦 BOX PLOT (Biểu đồ hộp)
   → So sánh phân phối giữa các nhóm
   → Dễ thấy outliers

📈 BAR CHART (Biểu đồ cột)
   → So sánh giá trị trung bình giữa các nhóm

🔵 SCATTER PLOT (Biểu đồ phân tán)
   → Xem mối quan hệ giữa 2 biến số

🔥 HEATMAP (Bản đồ nhiệt)
   → Xem tương quan giữa nhiều biến
```

---

### 6️⃣ Feature Engineering - Tạo biến mới

**Đây là nghệ thuật của Data Science!**

Tạo biến mới từ dữ liệu có sẵn để giúp mô hình học tốt hơn.

#### Biến 1: LOS_Group (Nhóm số ngày nằm viện)

```python
def categorize_los(los):
    if los <= 2:
        return 'Short (1-2 days)'
    elif los <= 5:
        return 'Medium (3-5 days)'
    elif los <= 10:
        return 'Long (6-10 days)'
    else:
        return 'Extended (>10 days)'

df['LOS_Group'] = df['Length of Stay'].apply(categorize_los)
```

**Tại sao tạo?**

- Thay vì 120 giá trị khác nhau (1, 2, 3, ..., 120)
- Ta nhóm thành 4 loại dễ hiểu hơn
- Mô hình có thể học pattern rõ ràng hơn

#### Biến 2: Urgency_Level (Mức độ khẩn cấp)

```python
def categorize_urgency(admission_type):
    if admission_type in ['Emergency', 'Trauma']:
        return 'High Urgency'
    elif admission_type == 'Urgent':
        return 'Medium Urgency'
    else:
        return 'Low Urgency'

df['Urgency_Level'] = df['Admission Type'].apply(categorize_urgency)
```

**Logic:**

- Cấp cứu/Chấn thương → Khẩn cấp cao → Chi phí cao
- Tự nguyện/Trẻ sơ sinh → Khẩn cấp thấp → Chi phí thấp hơn

#### Biến 3: Age_Risk (Rủi ro theo tuổi)

```python
def categorize_age_risk(age_group):
    if age_group in ['0 to 17', '70 or Older']:
        return 'High Risk Age'
    elif age_group in ['50 to 69']:
        return 'Medium Risk Age'
    else:
        return 'Low Risk Age'

df['Age_Risk'] = df['Age Group'].apply(categorize_age_risk)
```

**Logic:**

- Trẻ em và người già: Cần chăm sóc đặc biệt → Rủi ro cao
- Người trưởng thành: Khỏe mạnh hơn → Rủi ro thấp

---

### 7️⃣ Experiment - Thử nghiệm mô hình

#### Bước 1: Chuẩn bị dữ liệu

**Label Encoding** - Chuyển chữ thành số

```
Tại sao cần? Máy tính chỉ hiểu SỐ, không hiểu CHỮ!

TRƯỚC:                    SAU:
Age Group                 Age Group (encoded)
"0 to 17"      →          0
"18 to 29"     →          1
"30 to 49"     →          2
"50 to 69"     →          3
"70 or Older"  →          4
```

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Age_encoded'] = le.fit_transform(df['Age Group'])
```

#### Bước 2: Chia Train/Test

```python
from sklearn.model_selection import train_test_split

X = df[features]      # Các cột đầu vào
y = df['Total Charges']  # Cột cần dự đoán

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Tại sao phải chia?**

```
┌─────────────────────────────────────────────────────────────┐
│                     TOÀN BỘ DỮ LIỆU                         │
│                      (100% records)                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌───────────────────────────┐   ┌─────────────────────┐   │
│   │       TRAIN SET           │   │     TEST SET        │   │
│   │        (80%)              │   │      (20%)          │   │
│   │                           │   │                     │   │
│   │  Dùng để HUẤN LUYỆN      │   │  Dùng để ĐÁNH GIÁ  │   │
│   │  mô hình                  │   │  mô hình            │   │
│   │                           │   │                     │   │
│   │  "Đề cương ôn thi"        │   │  "Đề thi thật"      │   │
│   └───────────────────────────┘   └─────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Giống như học sinh:
- TRAIN = Làm bài tập về nhà (được biết đáp án)
- TEST = Làm bài kiểm tra (không biết đáp án trước)

Nếu chỉ dùng 1 bộ → Mô hình "học vẹt" → Gặp bài mới sẽ không làm được!
```

#### Bước 3: Feature Scaling (Chuẩn hóa dữ liệu)

**Vấn đề:**

```
Length of Stay: 1 - 120 (đơn vị: ngày)
Total Charges: 1,000 - 50,000 (đơn vị: $)

Số lớn sẽ "áp đảo" số nhỏ trong tính toán!
```

**Giải pháp - Scaling:**

| Scaler             | Cách hoạt động                            | Khi nào dùng     |
| ------------------ | ----------------------------------------- | ---------------- |
| **StandardScaler** | Chuyển về mean=0, std=1                   | Phổ biến nhất    |
| **MinMaxScaler**   | Chuyển về khoảng [0, 1]                   | Khi cần giới hạn |
| **RobustScaler**   | Dùng median, ít bị ảnh hưởng bởi outliers | Khi còn outliers |

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

#### Bước 4: Train các mô hình

Chúng ta thử **6 mô hình** với **3 scalers** = **18 combinations**!

##### Mô hình 1: Linear Regression (Hồi quy tuyến tính)

```
Ý tưởng: Tìm đường thẳng tốt nhất

y = a₁x₁ + a₂x₂ + ... + b

Chi phí = (hệ số × tuổi) + (hệ số × số ngày) + ... + hằng số
```

**Ưu điểm:** Đơn giản, nhanh, dễ hiểu
**Nhược điểm:** Chỉ bắt được quan hệ tuyến tính

##### Mô hình 2: Ridge Regression

```
Giống Linear Regression + "phạt" nếu hệ số quá lớn

Tránh overfitting (học vẹt)
```

##### Mô hình 3: Lasso Regression

```
Giống Ridge, nhưng có thể đẩy hệ số về 0

Tự động chọn features quan trọng
```

##### Mô hình 4: Decision Tree (Cây quyết định)

```
Ý tưởng: Chia dữ liệu theo các câu hỏi

                    Tuổi > 50?
                   /          \
                 Yes           No
                 /               \
         Nằm > 5 ngày?       Cấp cứu?
           /      \           /     \
         Yes      No        Yes     No
          |        |         |       |
       $40,000  $25,000   $20,000  $10,000
```

**Ưu điểm:** Dễ hiểu, giải thích được
**Nhược điểm:** Dễ overfitting

##### Mô hình 5: Random Forest (Rừng ngẫu nhiên)

```
Ý tưởng: Kết hợp NHIỀU cây quyết định

      Cây 1      Cây 2      Cây 3      ...      Cây 100
         ↓          ↓          ↓                   ↓
     $38,000    $42,000    $40,000             $39,000
                           ↓
              Trung bình: $40,000 (kết quả cuối)
```

**Ưu điểm:** Chính xác cao, khó overfit
**Nhược điểm:** Chậm, khó giải thích

##### Mô hình 6: Gradient Boosting

```
Ý tưởng: Xây dựng các cây TUẦN TỰ, cây sau sửa lỗi của cây trước

Cây 1: Dự đoán $30,000 (sai $10,000)
   ↓
Cây 2: Học từ sai số, điều chỉnh +$7,000
   ↓
Cây 3: Tiếp tục điều chỉnh +$2,000
   ↓
Kết quả: $39,000 (gần đúng hơn!)
```

**Ưu điểm:** Thường cho kết quả tốt nhất
**Nhược điểm:** Chậm, dễ overfit nếu không tune cẩn thận

---

### 8️⃣ Result - Đánh giá kết quả

#### Các chỉ số đánh giá

| Chỉ số       | Ý nghĩa                       | Công thức             | Tốt khi |
| ------------ | ----------------------------- | --------------------- | ------- |
| **R² Score** | % biến thiên được giải thích  | 1 - (SS_res / SS_tot) | Gần 1   |
| **RMSE**     | Sai số trung bình (căn bậc 2) | √(mean(y - ŷ)²)       | Nhỏ     |
| **MAE**      | Sai số trung bình tuyệt đối   | mean(\|y - ŷ\|)       | Nhỏ     |

**Giải thích R² bằng ví dụ:**

```
R² = 0.70 (70%)

Nghĩa là: Mô hình của chúng ta giải thích được 70% sự biến động của chi phí.
          30% còn lại phụ thuộc vào các yếu tố khác mà ta chưa đưa vào.
```

#### So sánh kết quả

```
┌────────────────────────┬──────────────┬─────────────┬─────────────┐
│ Model + Scaler         │ R² Score     │ RMSE ($)    │ MAE ($)     │
├────────────────────────┼──────────────┼─────────────┼─────────────┤
│ Random Forest + Robust │ 0.7123       │ $8,234      │ $5,421      │
│ Gradient Boost + Std   │ 0.7089       │ $8,312      │ $5,502      │
│ Decision Tree + MinMax │ 0.6845       │ $8,890      │ $5,823      │
│ Ridge + Standard       │ 0.5234       │ $10,234     │ $7,123      │
│ Linear + Standard      │ 0.5201       │ $10,289     │ $7,189      │
│ Lasso + Standard       │ 0.4892       │ $10,678     │ $7,456      │
└────────────────────────┴──────────────┴─────────────┴─────────────┘

🏆 Best Model: Random Forest + RobustScaler (R² = 0.71)
```

---

## 🖥️ Ứng dụng GUI Demo

### Mục đích

- Cho phép người dùng **nhập thông tin** và **nhận dự đoán**
- Demo cho thầy xem khi báo cáo

### Công nghệ

**Streamlit** - Framework tạo web app bằng Python

```
Streamlit biến code Python → Web App trong vài phút!
Không cần biết HTML, CSS, JavaScript!
```

### Cấu trúc ứng dụng

```
┌─────────────────────────────────────────────────────────────────┐
│                    🏥 NY HOSPITAL PREDICTOR                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────────┐  ┌──────────────────┐  │
│  │ Dự  │  │ EDA │  │Models│  │ Thống kê│  │Feature Engineering│  │
│  │đoán │  │     │  │      │  │         │  │                  │  │
│  └─────┘  └─────┘  └─────┘  └─────────┘  └──────────────────┘  │
│                                                                 │
│  Tab 1: Nhập thông tin bệnh nhân → Nhận dự đoán chi phí        │
│  Tab 2: Xem các biểu đồ phân tích                              │
│  Tab 3: So sánh 18 combinations Model + Scaler                 │
│  Tab 4: Xem thống kê dataset                                   │
│  Tab 5: Giải thích Feature Engineering                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Tab Dự đoán - Cách hoạt động

```
NGƯỜI DÙNG NHẬP:                     KẾT QUẢ:
─────────────────                    ────────
📍 Khu vực: New York City
🏥 Bệnh viện: NYU Hospital           ┌─────────────────────────┐
👤 Tuổi: 50-69                       │                         │
⚧ Giới tính: Nam              →     │  💰 Chi phí dự đoán:    │
🏃 Loại nhập viện: Emergency         │     $35,450.00          │
📅 Số ngày: 7 ngày                   │                         │
                                     └─────────────────────────┘
```

---

## 🚀 Cách chạy dự án

### Bước 1: Cài đặt môi trường

```bash
# Tạo môi trường ảo (khuyến khích)
python -m venv .venv

# Kích hoạt môi trường (Windows)
.venv\Scripts\activate

# Kích hoạt môi trường (Mac/Linux)
source .venv/bin/activate
```

### Bước 2: Cài đặt thư viện

```bash
pip install pandas numpy matplotlib seaborn scikit-learn plotly streamlit
```

### Bước 3: Chạy Notebook

1. Mở VS Code
2. Mở file `NY_Hospital_Charges_Analysis.ipynb`
3. Chọn kernel Python (từ .venv)
4. Chạy từng cell từ trên xuống

### Bước 4: Chạy GUI Demo

```bash
streamlit run gui_app.py
```

Trình duyệt sẽ tự động mở tại `http://localhost:8501`

---

## 📊 Kết quả và Kết luận

### Các phát hiện chính

1. **Số ngày nằm viện** là yếu tố ảnh hưởng MẠNH NHẤT đến chi phí
2. **Loại nhập viện** cấp cứu có chi phí cao hơn đáng kể
3. **Người già (70+)** và **trẻ em** có chi phí cao hơn
4. **Bệnh viện** khác nhau có mức giá rất khác nhau
5. **Random Forest** là mô hình tốt nhất với R² ≈ 0.71

### Hạn chế

- Chỉ có dữ liệu năm 2009 (có thể không còn phù hợp với hiện tại)
- Thiếu thông tin về loại bệnh/chẩn đoán
- R² = 0.71 nghĩa là vẫn còn 29% chưa giải thích được

### Hướng phát triển

- Thêm dữ liệu các năm khác
- Thêm thông tin chẩn đoán bệnh
- Thử các mô hình Deep Learning
- Deploy ứng dụng lên cloud (Heroku, AWS, ...)

---

## 📖 Thuật ngữ cần nhớ

| Thuật ngữ        | Tiếng Việt            | Giải thích                             |
| ---------------- | --------------------- | -------------------------------------- |
| **Dataset**      | Bộ dữ liệu            | Tập hợp dữ liệu để phân tích           |
| **Feature**      | Đặc trưng             | Các cột đầu vào (tuổi, giới tính, ...) |
| **Target**       | Mục tiêu              | Cột cần dự đoán (chi phí)              |
| **Train set**    | Tập huấn luyện        | Dữ liệu để dạy mô hình                 |
| **Test set**     | Tập kiểm tra          | Dữ liệu để đánh giá mô hình            |
| **Model**        | Mô hình               | Thuật toán học máy                     |
| **Prediction**   | Dự đoán               | Kết quả đầu ra của mô hình             |
| **Overfitting**  | Quá khớp              | Mô hình học vẹt, không tổng quát       |
| **Underfitting** | Chưa khớp             | Mô hình quá đơn giản, học chưa đủ      |
| **R² Score**     | Hệ số xác định        | Đo độ chính xác (0-1)                  |
| **RMSE**         | Sai số bình phương TB | Đo sai số dự đoán                      |
| **Scaler**       | Bộ chuẩn hóa          | Điều chỉnh scale của dữ liệu           |
| **Encoding**     | Mã hóa                | Chuyển chữ thành số                    |
| **Outlier**      | Ngoại lai             | Giá trị bất thường                     |
| **EDA**          | Phân tích khám phá    | Tìm hiểu dữ liệu ban đầu               |
| **Pipeline**     | Quy trình             | Các bước xử lý tuần tự                 |

---

## ❓ FAQ - Câu hỏi thường gặp

### Q: Tại sao chọn Random Forest làm best model?

**A:** Vì nó cho R² cao nhất (0.71) và RMSE thấp nhất. Random Forest kết hợp nhiều cây quyết định, giảm overfitting và ổn định hơn.

### Q: Tại sao cần Feature Engineering?

**A:** Để giúp mô hình học tốt hơn. Ví dụ, thay vì để mô hình tự hiểu "70 tuổi = rủi ro cao", ta tạo sẵn biến Age_Risk để mô hình dễ học hơn.

### Q: R² = 0.71 có tốt không?

**A:** Khá tốt cho bài toán dự đoán chi phí y tế! Chi phí y tế phụ thuộc vào nhiều yếu tố mà ta không có dữ liệu (loại bệnh, biến chứng, bảo hiểm, ...). 0.71 nghĩa là ta giải thích được 71% biến động.

### Q: Tại sao loại bỏ outliers?

**A:** Outliers (chi phí $4 triệu) làm mô hình bị lệch. Đa số bệnh nhân chi phí $10,000-$50,000, nên ta tập trung dự đoán cho đa số này.

### Q: Streamlit là gì?

**A:** Framework Python để tạo web app dễ dàng. Chỉ cần viết Python, không cần biết web development (HTML, CSS, JS).

---

## 👨‍💻 Thông tin liên hệ

Nếu có thắc mắc về dự án, liên hệ thành viên trong nhóm!

---

<div align="center">

**🎓 Machine Learning - Final Exam Project**

_Phân tích và Dự đoán Chi phí Nhập viện tại các Bệnh viện Bang New York (2009)_

</div>
