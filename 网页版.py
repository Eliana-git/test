import pandas as pd
import numpy as np
import re
import jieba
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

# 确保中文分词库有合适的环境
jieba.setLogLevel(jieba.logging.INFO)

# 文本预处理函数
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', 'num', text)
    text = text.lower()
    return text

# 读取数据函数 - 修改为提取文字标签
def read_data(file_path):
    data = []
    # 创建数字标签到文字标签的映射
    label_id_to_text = {
        '100': '民生 故事',
        '101': '文化',
        '102': '娱乐',
        '103': '体育',
        '104': '财经',
        '106': '房产',
        '107': '汽车',
        '108': '教育',
        '109': '科技',
        '110': '军事',
        '112': '旅游',
        '113': '国际',
        '114': '证券 股票',
        '115': '农业 三农',
        '116': '电竞 游戏'
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('_!_')
            if len(parts) == 5:
                numeric_label = parts[1]  # 数字标签
                text_label = parts[2]     # 文字标签
                
                # 提取文字标签的简短形式
                short_text_label = text_label.split('_')[-1] if '_' in text_label else text_label
                
                title = parts[3]
                words = jieba.cut(clean_text(title))
                processed_text = ' '.join(words)
                data.append([processed_text, numeric_label])  # 使用数字标签进行训练
    
    df = pd.DataFrame(data, columns=['text', 'label'])
    return df, label_id_to_text

# 数据预处理和特征工程
def preprocess_data(df):
    X = df['text']
    y = df['label']
    
    # 创建标签到数字的映射
    unique_labels = sorted(y.unique())
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for i, label in enumerate(unique_labels)}
    
    y = y.map(label_to_id)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
        max_features=10000,
        use_idf=True,
        smooth_idf=True
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 特征选择
    selector = SelectKBest(chi2, k=5000)
    X_train_vec = selector.fit_transform(X_train_vec, y_train)
    X_test_vec = selector.transform(X_test_vec)
    
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer, selector, id_to_label

# 模型训练
def train_model(X_train, y_train):
    model = lgb.LGBMClassifier(
        boosting_type='gbdt',
        objective='multiclass',
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    return model

# 模型评估
def evaluate_model(model, X_test, y_test, id_to_label, label_id_to_text):
    y_pred = model.predict(X_test)
    
    # 将数字ID转回数字标签，再转回文字标签
    y_test_labels = [label_id_to_text[id_to_label[i]] for i in y_test]
    y_pred_labels = [label_id_to_text[id_to_label[i]] for i in y_pred]
    
    accuracy = accuracy_score(y_test_labels, y_pred_labels)
    precision = precision_score(y_test_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_test_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_test_labels, y_pred_labels, average='weighted')
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 预测函数
def predict_label(text, model, vectorizer, selector, id_to_label, label_id_to_text):
    # 预处理输入文本
    processed_text = clean_text(text)
    words = jieba.cut(processed_text)
    processed_text = ' '.join(words)
    
    # 特征向量化
    try:
        text_vector = vectorizer.transform([processed_text])
        text_vector = selector.transform(text_vector)
        
        # 预测
        prediction = model.predict(text_vector)[0]
        
        # 映射回数字标签
        numeric_label = id_to_label[prediction]
        
        # 映射到文字标签
        return label_id_to_text.get(numeric_label, "未知分类")
    except Exception as e:
        print(f"预测错误: {e}")
        return "未知分类"

# 创建Flask应用
app = Flask(__name__)

# 全局变量
model = None
vectorizer = None
selector = None
id_to_label = None
label_id_to_text = None
metrics = None

# 首页路由
@app.route('/')
def index():
    return render_template('index.html')

# API路由：预测分类
@app.route('/predict', methods=['POST'])
def predict():
    global model, vectorizer, selector, id_to_label, label_id_to_text
    
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': '请提供文本内容'}), 400
        
        # 预测标签
        predicted_label = predict_label(text, model, vectorizer, selector, id_to_label, label_id_to_text)
        
        return jsonify({'label': predicted_label})
    
    except Exception as e:
        print(f"预测错误: {e}")
        return jsonify({'error': '预测过程中发生错误'}), 500

# API路由：获取模型指标
@app.route('/metrics', methods=['GET'])
def get_metrics():
    global metrics
    
    try:
        if metrics:
            return jsonify(metrics)
        else:
            return jsonify({
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0
            })
    except Exception as e:
        print(f"获取指标错误: {e}")
        return jsonify({
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }), 500

# 主函数
if __name__ == "__main__":
    file_path = 'toutiao_cat_data.txt'
    print("正在加载数据...")
    df, label_id_to_text = read_data(file_path)
    
    print("正在预处理数据...")
    X_train_vec, X_test_vec, y_train, y_test, vectorizer, selector, id_to_label = preprocess_data(df)
    
    print("正在训练模型...")
    model = train_model(X_train_vec, y_train)
    
    print("正在评估模型...")
    metrics = evaluate_model(model, X_test_vec, y_test, id_to_label, label_id_to_text)
    
    print("启动Web应用...")
    print("请访问 http://localhost:5000 使用文本分类功能")
    app.run(debug=True, use_reloader=False)    