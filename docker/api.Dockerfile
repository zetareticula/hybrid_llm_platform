FROM rust:1.76 as builder
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates
WORKDIR /app
COPY --from=builder /app/target/release/hybrid-llm-api .
EXPOSE 8080
CMD ["./hybrid-llm-api"]