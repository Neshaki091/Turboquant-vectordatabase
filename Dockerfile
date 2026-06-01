# Giai đoạn 1: Builder
FROM rust:1.76-slim as builder
WORKDIR /usr/src/turboquant

# Copy mã nguồn
COPY . .

# Biên dịch tối ưu hóa Release
RUN cargo build --release

# Giai đoạn 2: Runtime (Siêu nhẹ)
FROM debian:bookworm-slim
WORKDIR /app

# Chỉ copy file nhị phân (Binary) duy nhất từ Builder
COPY --from=builder /usr/src/turboquant/target/release/turboquant-server /app/turboquant-server

# Mở cổng 6333
EXPOSE 6333

# Chạy Server
CMD ["./turboquant-server"]
