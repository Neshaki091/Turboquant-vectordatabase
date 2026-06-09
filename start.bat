@echo off
echo ==============================================
echo Khởi động TurboQuant Standalone Server (Rust)
echo ==============================================
set RUSTFLAGS=-C target-cpu=native
cargo run --release
pause
