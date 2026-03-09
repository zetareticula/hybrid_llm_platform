use tokio::sync::broadcast;
use axum::{
    extract::{State, ws::WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use axum::extract::ws::{Message as WsMessage, WebSocket};

/// Initialise the broadcast channel used to push metrics events to WebSocket subscribers.
pub fn init_metrics_channel() -> broadcast::Sender<String> {
    let (tx, _) = broadcast::channel(256);
    tx
}

/// Start the WebSocket metrics push server on port 8081.
pub async fn start_ws_server(tx: broadcast::Sender<String>) {
    let app = Router::new()
        .route("/metrics", get(ws_handler))
        .with_state(tx);

    axum::Server::bind(&"0.0.0.0:8081".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn ws_handler(
    State(tx): State<broadcast::Sender<String>>,
    ws: WebSocketUpgrade,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, tx))
}

async fn handle_socket(mut socket: WebSocket, tx: broadcast::Sender<String>) {
    let mut rx = tx.subscribe();
    loop {
        match rx.recv().await {
            Ok(msg) => {
                if socket.send(WsMessage::Text(msg)).await.is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}
