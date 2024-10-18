# Flowchart

- Người dùng tải lên file
- Hệ thống UI/API xử lý và gửi file lên hệ thống Sandbox CAPEv2 (Cuckoo host) để phân tích thông qua API send POST request
- Hệ thống Cuckoo host nhận file và thực hiện phân tích như sau
  - Cuckoo host thực hiện các bước static analysis như: 
    - Kiểm tra file type
    - Kiểm tra file size
    - Kiểm tra file hash
    - Kiểm tra file signature
    - Kiểm tra file metadata
    - ...
  - Cuckoo host thông qua socket server để thực hiện gọi đến VirtualBox (VBoxManage) để restore snapshot/khởi động máy ảo đã được cấu hình (Guest).
  - Cuckoo host thực hiện gửi file đến Guest, khởi chạy file và ghi lại các hành vi của file thông qua một agent (Cuckoo agent) đang chạy trên máy ảo. (Giao tiếp thông qua Virtual Network)
  - Agent sau khi đã thu thập được các thông tin hành vi của file sẽ gửi về Cuckoo host để xử lý. (Giao tiếp thông qua Virtual Network)
- Cuckoo host sau khi nhận được các thông tin hành vi của file và kèm với các thông tin từ static analysis sẽ thực hiện thêm các bước phân tích như:
  - Kiểm tra với VirusTotal
  - Kiểm tra với Yara
  - Kiểm tra với Suricata
- Cuckoo host sau khi đã hoàn thành phân tích sẽ có các report dạng json, HTML, PDF. Hệ thống sẽ lưu trữ các report này và gửi report json đến hệ thống Machine Learning để xử lý và đưa ra kết quả đánh giá file.
- Hệ thống Machine Learning sau khi đánh giá xong sẽ gửi kết quả đánh giá về hệ thống UI/API để hiển thị kết quả cho người dùng và kèm theo PDF report từ Cuckoo host.