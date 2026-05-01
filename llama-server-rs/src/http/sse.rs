use crate::engine::task::TaskResult;
use async_stream::stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use tokio::sync::broadcast;

pub fn stream_task_events(
    mut rx: broadcast::Receiver<TaskResult>,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let stream = stream! {
        while let Ok(event) = rx.recv().await {
            match event {
                TaskResult::Chunk(v) => {
                    yield Ok(Event::default().event("chunk").data(v.to_string()));
                }
                TaskResult::Done(v) => {
                    yield Ok(Event::default().event("done").data(v.to_string()));
                    break;
                }
                TaskResult::Error(e) => {
                    yield Ok(Event::default().event("error").data(e));
                    break;
                }
            }
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}
