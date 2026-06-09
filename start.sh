#!/bin/bash
echo "🚀 Khởi động quá trình biên dịch TurboQuant Server trên Linux..."

# Cài đặt Rust nếu chưa có
if ! command -v cargo &> /dev/null; then
    echo "Rust chưa được cài đặt. Đang tải và cài đặt Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

echo "Đang biên dịch lõi TurboQuant (Release Mode - Tối ưu hóa AVX)..."
RUSTFLAGS="-C target-cpu=native" cargo build --release

echo "====================================="
echo "✅ Biên dịch thành công!"
echo "Đang chạy Server..."
./target/release/turboquant-server
