# Import các thư viện cần thiết
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy
import re
import underthesea # Thư viện tách từ

from sklearn.model_selection import train_test_split # Thư viện chia tách dữ liệu

from transformers import AutoModel, AutoTokenizer # Thư viện BERT

# Thư viện train SVM
from sklearn.svm import SVC
from joblib import dump


# Hàm load model BERT
def load_bert():
    v_phobert = AutoModel.from_pretrained("vinai/phobert-base")
    v_tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    return v_phobert, v_tokenizer

# Hàm chuẩn hoá câu
def standardize_data(row):
    # Xóa dấu chấm, phẩy, hỏi ở cuối câu
    row = re.sub(r"[\.,\?]+$-", "", row)
    # Xóa tất cả dấu chấm, phẩy, chấm phẩy, chấm thang, ... trong câu
    row = row.replace(",", " ").replace(".", " ") \
        .replace(";", " ").replace("“", " ") \
        .replace(":", " ").replace("”", " ") \
        .replace('"', " ").replace("'", " ") \
        .replace("!", " ").replace("?", " ") \
        .replace("-", " ").replace("?", " ")
    row = row.strip().lower()
    return row

# Hàm load danh sách các từ vô nghĩa: lắm, ạ, à, bị, vì..
def load_stopwords():
    sw = []
    with open("stopwords.txt", encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        sw.append(line.replace("\n",""))
    return sw


# Hàm load dữ liệu từ file data_1.csv để train model
def load_data():
    v_text = []
    v_label = []

    with open('data_1.csv', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.replace("\n","")
        print(line[:-2])
        v_text.append(standardize_data(line[:-2]))
        v_label.append(int(line[-1:].replace("\n", "")))

    print(v_label)
    return v_text, v_label


# Hàm tạo ra bert features
def make_bert_features(v_text):
    global phobert, sw
    v_tokenized = []
    max_len = 100 # Mỗi câu dài tối đa 100 từ
    for i_text in v_text:
        print("Đang xử lý line = ", i_text)
        # Phân thành từng từ
        line = underthesea.word_tokenize(i_text)
        # Lọc các từ vô nghĩa
        filtered_sentence = [w for w in line if not w in sw]
        # Ghép lại thành câu như cũ sau khi lọc
        line = " ".join(filtered_sentence)
        line = underthesea.word_tokenize(line, format="text")
        # print("Word segment  = ", line)
        # Tokenize bởi BERT
        line = tokenizer.encode(line)
        v_tokenized.append(line)

    # Chèn thêm số 1 vào cuối câu nếu như không đủ 100 từ
    padded = numpy.array([i + [1] * (max_len - len(i)) for i in v_tokenized])
    print('padded:', padded[0])
    print('len padded:', padded.shape)

    # Đánh dấu các từ thêm vào = 0 để không tính vào quá trình lấy features
    attention_mask = numpy.where(padded == 1, 0, 1)
    print('attention mask:', attention_mask[0])

    # Chuyển thành tensor
    padded = torch.tensor(padded).to(torch.long)
    print("Padd = ",padded.size())
    attention_mask = torch.tensor(attention_mask)

    # Lấy features dầu ra từ BERT
    with torch.no_grad():
        last_hidden_states = phobert(input_ids= padded, attention_mask=attention_mask)

    v_features = last_hidden_states[0][:, 0, :].numpy()
    print(v_features.shape)
    return v_features


print("Chuẩn bị nạp danh sách các từ vô nghĩa (stopwords)...")
sw = load_stopwords()
print("Đã nạp xong danh sách các từ vô nghĩa")

print("Chuẩn bị nạp model BERT....")
phobert, tokenizer = load_bert()
print("Đã nạp xong model BERT.")

print("Chuẩn bị load dữ liệu....")
text, label = load_data()
print("Đã load dữ liệu xong")

print("Chuẩn bị tạo features từ BERT.....")
features = make_bert_features(text)
print("Đã tạo xong features từ BERT")

# Phân chia dữ liệu train, test
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)

#
# parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 2, 4], 'gamma': [0.125, 0.25, 0.5, 1, 2, 4]}
# from sklearn.model_selection import GridSearchCV
# clf = GridSearchCV(SVC(), param_grid=parameters)
# grid_search = clf.fit(X_train, y_train)
#
# print("Best score: %0.3f" % grid_search.best_score_)
# print(grid_search.best_estimator_)
#
# # best prarams
# print('best prarams:', clf.best_params_)

print("Chuẩn bị train model SVM....")
cl = SVC(kernel='linear', probability=True, gamma=0.125)
cl.fit(features, label)

sc = cl.score(X_test, y_test)
print('Kết quả train model, độ chính xác = ', sc*100, '%')

# Save model
dump(cl, 'save_model.pkl')
print("Đã lưu model SVM vào file save_model.pkl")
