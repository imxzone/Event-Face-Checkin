# Event Face Check-in System for Edu Campus
Hệ thống điểm danh sự kiện bằng nhận diện khuôn mặt, ứng dụng AI cho môi trường giáo dục tại FPT University.

---

## 1. Giới thiệu (Overview)

**Event Face Check-in System** là một hệ thống điểm danh tự động sử dụng công nghệ **nhận diện khuôn mặt (Face Recognition)**, được thiết kế nhằm phục vụ các sự kiện học thuật và ngoại khóa tại **FPT University Edu Campus**.

Hệ thống giúp thay thế các phương pháp điểm danh thủ công hoặc QR code truyền thống, từ đó:
- Rút ngắn thời gian check-in
- Hạn chế gian lận điểm danh hộ
- Tự động hóa việc ghi nhận và quản lý dữ liệu tham dự

Project được xây dựng theo hướng **Applied AI System**, tập trung vào tính ứng dụng thực tế trong môi trường giáo dục.

---

## 2. Bài toán & Động lực (Problem Statement)

Trong các sự kiện tại trường đại học, công tác điểm danh thường gặp các vấn đề:
- Tốn nhiều nhân lực vận hành
- Dễ xảy ra sai sót khi nhập liệu thủ công
- QR code có thể bị chia sẻ hoặc check-in hộ
- Khó tổng hợp dữ liệu theo thời gian thực

➡️ **Mục tiêu của project** là xây dựng một hệ thống AI có thể:
- Nhận diện người tham dự nhanh chóng, không tiếp xúc
- Mỗi người chỉ được check-in một lần
- Tự động lưu lại dữ liệu điểm danh

---

## 3. Tổng quan giải pháp (Solution Overview)

Hệ thống sử dụng camera để thu nhận hình ảnh khuôn mặt người tham dự theo thời gian thực.  
Dữ liệu khuôn mặt sẽ được xử lý và so khớp với dữ liệu đã đăng ký trước đó để xác định danh tính.

Khi nhận diện thành công:
- Trạng thái check-in được cập nhật
- Thời gian check-in được ghi nhận
- Dữ liệu được lưu dưới dạng CSV / JSON hoặc gửi qua API

---

## 4. Kiến trúc hệ thống (System Architecture)

Quy trình hoạt động tổng quát của hệ thống:

1. Camera thu hình khuôn mặt theo thời gian thực
2. Phát hiện và căn chỉnh khuôn mặt (Face Detection & Alignment)
3. Trích xuất vector đặc trưng khuôn mặt (Face Embedding)
4. So khớp với dữ liệu đã lưu bằng cosine similarity
5. Kiểm tra ngưỡng nhận diện và trạng thái check-in
6. Ghi nhận kết quả điểm danh

---

## 5. Tính năng chính (Key Features)

- Nhận diện khuôn mặt theo thời gian thực
- Mỗi người chỉ được check-in một lần
- Thời gian chờ (cooldown) để tránh check-in lặp
- Lưu dữ liệu điểm danh tự động (CSV / JSON)
- Hỗ trợ chia sẻ dữ liệu giữa nhiều máy thông qua API
- Cấu hình linh hoạt ngưỡng nhận diện và thời gian cooldown

---

## 6. Công nghệ sử dụng (Tech Stack)

- **Ngôn ngữ:** Python 3.10  
- **AI / Computer Vision:** InsightFace  
- **Xử lý ảnh & camera:** OpenCV  
- **Xử lý dữ liệu:** NumPy, Pandas  
- **Backend / API:** Flask  
- **Lưu trữ:** CSV, JSON  

---

## 7. Cấu trúc thư mục (Project Structure)

```text
Event-Face-Checkin/
├── data/
│   ├── raw/                # Ảnh gốc ban đầu
│   ├── faces/              # Ảnh khuôn mặt đã crop
│   ├── processed/          # Embeddings và log check-in
├── src/
│   ├── preprocess.py       # Xử lý dữ liệu khuôn mặt
│   ├── main.py             # Chạy hệ thống check-in
│   ├── api.py              # API chia sẻ dữ liệu
│   ├── config.py           # File cấu hình hệ thống
├── ui/
│   └── main.ui             # Giao diện người dùng (Qt)
├── requirements.txt
└── README.md
