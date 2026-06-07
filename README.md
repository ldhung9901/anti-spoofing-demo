# Attendance Liveness WebSocket Demo

Demo chấm công kiểu eKYC:

- HTML browser mở camera.
- Gửi frame JPEG qua WebSocket về .NET 8 backend.
- .NET giữ session + active challenge random.
- .NET gọi Python AI worker chạy CPU.
- Python worker dùng MediaPipe FaceMesh để lấy pose/blink và MiniFASNet-V2 ONNX từ Hugging Face để anti-spoofing.

> Đây là demo kỹ thuật, chưa phải bank-grade PAD. Production cần calibrate threshold, test camera thật, log risk, chống replay, device binding, HTTPS/WSS, và đánh giá PAD nghiêm túc.

## Cấu trúc

```txt
AttendanceLivenessWsDemo/
  src/AttendanceLivenessDemo.Api/
    Program.cs
    appsettings.json
    wwwroot/index.html
  ai-worker/
    app.py
    requirements.txt
    models/                 # model ONNX sẽ được tải vào đây
  scripts/
    run-all-windows.ps1
    run-ai-worker.ps1
    run-dotnet-api.ps1
```

## Model Hugging Face

Mặc định worker dùng repo:

```txt
garciafido/minifasnet-v2-anti-spoofing-onnx
```

Worker sẽ tự tìm file `.onnx` đầu tiên trong repo và tải vào `ai-worker/models` khi start lần đầu.

Có thể override:

```powershell
$env:HF_REPO_ID="garciafido/minifasnet-v2-anti-spoofing-onnx"
$env:HF_MODEL_FILE="ten-file.onnx"
```

Nếu production server không có internet, chạy worker một lần ở máy dev để tải model, sau đó copy file `.onnx` vào `ai-worker/models` trên server.

## Chạy trên Windows

Yêu cầu:

- .NET SDK 8
- Python 3.11 khuyến nghị cho MediaPipe
- Camera browser

Chạy tất cả:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\run-all-windows.ps1
```

Mở:

```txt
http://localhost:5088
```

## Chạy từng service

Terminal 1:

```powershell
.\scripts\run-ai-worker.ps1
```

Terminal 2:

```powershell
.\scripts\run-dotnet-api.ps1
```

## Linux/macOS

```bash
./scripts/run-ai-worker.sh
./scripts/run-dotnet-api.sh
```

## Test API

AI worker:

```txt
http://127.0.0.1:8001/health
```

.NET:

```txt
http://localhost:5088/health
```

## Tối ưu CPU

Trong `ai-worker/app.py`:

```txt
ORT_THREADS=2
```

Nếu server yếu, giảm FPS ở `wwwroot/index.html`:

```js
const uploadIntervalMs = 250; // 4 fps
```

Hoặc giảm canvas:

```html
<canvas id="canvas" width="256" height="192"></canvas>
```

Khuyến nghị demo:

- 5–6 fps
- JPEG quality 0.65–0.70
- 320×240
- 1 worker Python/process nếu CPU yếu

## Luồng active challenge

Backend sinh random:

```txt
CENTER_FACE + [TURN_LEFT, TURN_RIGHT, BLINK_TWICE] random order
```

Pass khi:

- Thấy đúng 1 mặt.
- Ảnh đủ chất lượng.
- Yaw đạt ngưỡng khi quay trái/phải.
- EAR giảm đủ để tính blink.
- Passive anti-spoof model không báo spoof risk quá cao.

## Ghi chú triển khai Flutter/WebView

WebView có thể mở URL demo này và vẫn gửi frame về backend. Nhưng production nên dùng Flutter native camera để ổn định hơn. Backend vẫn phải là nơi quyết định kết quả cuối, không tin `true/false` từ client.
