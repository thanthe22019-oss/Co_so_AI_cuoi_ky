# Bài tập lớn Pac-Man — CS188 Spring 2026

Đã chọn phần mở rộng: so sánh Q-Learning dạng bảng, xấp xỉ tuyến tính gốc, đặc trưng cải tiến và ablation trên mê cung 15×15, 25×25, 35×35. Mốc không học là ReflexAgent.

## Mở báo cáo

- `report/Bao_cao_PacMan.docx`: báo cáo Word, có bảng số liệu và hình.
- `report/bao_cao.md`: nguồn báo cáo tiếng Việt.
- `results/main/episodes.csv`: từng ván thực nghiệm chính.
- `results/main/per_seed.csv`, `summary.csv`: tổng hợp theo seed và thuật toán.
- `logs/project1.txt`, `project2.txt`, `project3.txt`, `project5.txt`: bằng chứng autograder.

Bổ sung tên nhóm, thành viên, mã sinh viên, lớp và giảng viên trước khi nộp. Các kết quả là kết quả chạy thật trên máy hiện tại, không phải cam kết điểm của giảng viên.

## Môi trường

Phiên hiện tại đã có `.venv` với PyTorch CPU, dùng numpy và matplotlib từ Python hệ thống. Python 3.14 đã chạy được bộ kiểm tra. Nếu chuyển máy, tạo môi trường mới (không sao chép `.venv`):

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install numpy matplotlib python-docx
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dùng phiên bản Python mà PyTorch cung cấp wheel tương ứng. `results/environment.txt` ghi các phiên bản thực tế của lần chạy này. Các lệnh dưới đây chạy từ thư mục gốc bài tập. `-X utf8` giúp xử lý đường dẫn tiếng Việt.

## Kiểm tra

```powershell
python -X utf8 scripts/check_projects.py
python -X utf8 -m unittest discover -s tests -v
```

Có thể chạy riêng: `python -X utf8 scripts/check_projects.py --projects 1 3`. Script đọc điểm và chuỗi FAIL vì autograder Berkeley có thể trả exit code 0 dù một số câu chưa đạt. Chạy lại sẽ ghi đè log tương ứng.

## Demo các project

Mỗi project có các module trùng tên; phải chạy đúng thư mục và không trộn mã giữa chúng.

Project 1 — A* đi qua bốn góc:

```powershell
Push-Location projects/project1/search
python -X utf8 pacman.py -l mediumCorners -p AStarCornersAgent
Pop-Location
```

Project 2 — Expectimax và hàm đánh giá cải tiến:

```powershell
Push-Location projects/project2/multiagent
python -X utf8 pacman.py -l smallClassic -p ExpectimaxAgent -a evalFn=better,depth=2 -k 1
Pop-Location
```

Project 3 — Value Iteration, Q-Learning bảng và Q-Learning xấp xỉ:

```powershell
Push-Location projects/project3/reinforcement
python -X utf8 gridworld.py -a value -i 100
python -X utf8 pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid -q
python -X utf8 pacman.py -p ApproximateQAgent -a extractor=SimpleExtractor -x 50 -n 60 -l mediumGrid
Pop-Location
```

Bỏ `-q` hoặc thêm các lựa chọn hiển thị phù hợp để xem trò chơi; các ván huấn luyện được engine tự ẩn. Project 5 chạy kiểm tra/huấn luyện riêng:

```powershell
python -X utf8 scripts/check_projects.py --projects 5
```

Kết quả học mạng có ngẫu nhiên; ngưỡng dừng và kiến trúc được mô tả trong báo cáo. Câu character-GPT không có điểm và không thuộc autograder phân phối; chưa triển khai phần tự chọn này.

## Thí nghiệm mở rộng

Chạy nhanh để kiểm tra:

```powershell
python -X utf8 experiments/benchmark.py --sizes 15 --seeds 0 --budget 500 --eval-games 3 --max-steps 100 --output results/quick
```

Tái lập cấu hình chính (có thể mất nhiều phút trên CPU):

```powershell
python -X utf8 experiments/benchmark.py --budget 10000 --eval-games 100 --max-steps 500 --output results/reproduction
python -X utf8 experiments/summarize.py --input results/reproduction
```

`summarize.py` cần numpy, matplotlib và python-docx; nếu dùng máy mới nên chạy bằng `.venv/Scripts/python.exe`. Lệnh tổng hợp ghi đè báo cáo và các biểu đồ trong `report/`. Giữ dữ liệu `results/main` để đối chiếu. Không chạy hai benchmark vào cùng thư mục output cùng lúc.

Phiên này dùng `python -X utf8 experiments/parallel.py --workers 8` cho các cấu hình độc lập, giữ nguyên ngân sách và seed. Công cụ đọc `results/main/config.json`, tái sử dụng lượt hoàn chỉnh và ghi mỗi lượt mới vào `results/shards/`. Dừng các benchmark cũ trước khi chạy công cụ này. `results/main/execution.json` ghi cách thực thi; thời gian đo chịu ảnh hưởng của chia sẻ CPU.

Sau khi hoàn tất, chạy `python -X utf8 scripts/verify_results.py` để xác nhận đủ cấu hình, đủ số ván và đúng ngân sách học.

Nếu gián đoạn, thêm `--resume` vào đúng lệnh ban đầu. Chương trình yêu cầu cấu hình khớp, giữ các lượt đã đánh giá đủ số ván và đủ ngân sách, rồi chạy lại lượt chưa hoàn tất. Bản CSV trước lần tiếp tục đầu được giữ tại `episodes-before-resume.csv`.

Các layout được lưu trong `experiments/layouts`. Engine giữ một ma ngẫu nhiên, hai capsule và phần thưởng gốc. 500 bước là timeout, không phải thắng/thua nhân tạo. Đánh giá không học. Các phương pháp học dùng đúng ngân sách tương tác bằng nhau; không so công bằng bằng số episode vì episode có độ dài khác nhau.

Reflex không có quá trình học và runner chỉ chạy seed đầu của danh sách. Các vòng lặp seed Reflex trùng nhau từ lần chạy ban đầu được loại khi tổng hợp; không coi các bản sao là mẫu độc lập. Phần báo cáo trình bày cả những kết quả không thắng và giới hạn của ngân sách nhỏ. Không có checkpoint mô hình được lưu; cần huấn luyện lại khi tái lập.

## Nguồn và ghi nhận

Mã nền và autograder: UC Berkeley, John DeNero, Dan Klein và cộng sự, https://inst.eecs.berkeley.edu/~cs188/sp26/projects/ . `projects/sources.json` lưu URL ZIP và SHA256. Giữ nguyên thông báo bản quyền trong từng file. Không công khai bộ lời giải lên kho công cộng theo điều kiện sử dụng của mã gốc.

Kiến thức học phần: bốn PDF người dùng cung cấp. Hướng RL dựa chủ yếu vào `12_RL.pdf`, kết hợp MDP từ `11_MDP.pdf`. Project 5 sử dụng kiến thức mạng nơron từ `14_neural network.pdf`.
