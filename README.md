# An Explainable Framework for Fine-Grained Sentiment Analysis of Vietnamese E-commerce Reviews

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Models-yellow.svg)](https://huggingface.co/DucAnh204/vietnamese-phobert-sentiment-absa)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face%20Datasets-yellow.svg)](https://huggingface.co/DucAnh204/vietnamese-skincare-reviews)

Khung làm việc hỗ trợ phân tích cảm xúc chi tiết (Fine-grained Sentiment Analysis) và đánh giá khía cạnh (ABSA) kết hợp XAI giải thích quyết định mô hình dành cho các bài viết đánh giá mỹ phẩm tiếng Việt trên sàn thương mại điện tử Shopee.

---

## 🚀 Hugging Face Assets

Toàn bộ tài nguyên mô hình và tập dữ liệu nặng (đã được lược bỏ khỏi lịch sử Git) có sẵn tại các link chính thức dưới đây:

* **📦 Tập dữ liệu đồ án (Dataset):** [DucAnh204/vietnamese-skincare-reviews](https://huggingface.co/DucAnh204/vietnamese-skincare-reviews)
  * Chứa file gốc cào từ Shopee (`shopee_reviews_merged.csv`), tập dữ liệu đã qua tiền xử lý chuẩn hóa (`reviews_final_cleaned.csv`), và các phân mục dữ liệu huấn luyện/kiểm thử.
* **🧠 Mô hình đã huấn luyện (Model Weights):** [DucAnh204/vietnamese-phobert-sentiment-absa](https://huggingface.co/DucAnh204/vietnamese-phobert-sentiment-absa)
  * Chứa trọng số mô hình PhoBERT ABSA (3 lớp cân bằng), PhoBERT Rating Predictor (1-5 sao), và mô hình cơ sở CNN text-only.

---

## 📁 Cấu trúc thư mục dự án

```text
/home/anhvd/DATN/
├── src/                # Mã nguồn ứng dụng và logic xử lý chính
│   ├── crawl/          # Code cào dữ liệu Shopee bằng Playwright
│   ├── preprocess/     # Pipeline tiền xử lý và lọc thư rác
│   ├── processing/     # Chia phân mục dữ liệu cho ABSA và Rating
│   ├── training/       # Code huấn luyện mô hình PhoBERT & CNN
│   ├── inference/      # Engine suy luận tích hợp XAI (Integrated Gradients & Attention)
│   └── web.py          # File khởi chạy giao diện Web UI chính
├── paper/              # Chứa tài liệu bài báo cáo đồ án sạch bôi vàng
│   └── paper.tex       # File mã nguồn LaTeX
├── requirements.txt    # Danh sách các thư viện Python phụ thuộc
└── .gitignore          # Cấu hình bỏ qua các thư mục dữ liệu/model nặng cục bộ
```

---

## 🔧 Hướng dẫn cài đặt và thiết lập nhanh

### 1. Clone dự án và Cài đặt thư viện
```bash
git clone git@github.com:BearBear204/phan_tich_cam_xuc_chi_tiet_co_giai_thich_tmdt_vn.git
cd phan_tich_cam_xuc_chi_tiet_co_giai_thich_tmdt_vn
pip install -r requirements.txt
```

### 2. Tải về Dữ liệu và Mô hình
Để ứng dụng hoạt động chính xác, bạn cần tải về thư mục `data/` và `models/` từ Hugging Face và đặt trực tiếp tại thư mục gốc của dự án:
```bash
# Cài đặt huggingface-cli nếu chưa có
pip install huggingface_hub

# Tải và thiết lập thư mục model
huggingface-cli download DucAnh204/vietnamese-phobert-sentiment-absa --local-dir models

# Tải và thiết lập thư mục data
huggingface-cli download DucAnh204/vietnamese-skincare-reviews --local-dir data --repo-type dataset
```

---

## 🖥️ Hướng dẫn sử dụng giao diện Web UI

Khởi động server Flask cho giao diện web tương tác phân tích và giải thích:
```bash
python src/web.py
```

Sau khi khởi chạy thành công, mở trình duyệt và truy cập:
👉 **[http://127.0.0.1:5002](http://127.0.0.1:5002)**

Giao diện hỗ trợ:
1. Nhập review mỹ phẩm tiếng Việt bất kỳ.
2. Dự đoán số sao đánh giá (1-5★).
3. Đánh giá cảm xúc khía cạnh (ABSA) đối với 3 thành phần: *Chất lượng sản phẩm*, *Dịch vụ người bán*, và *Dịch vụ giao hàng*.
4. Hiển thị bản đồ nhiệt (Heatmap) giải thích các từ khóa đóng góp chính bằng **Integrated Gradients** hoặc **Attention-based visualizer**.
