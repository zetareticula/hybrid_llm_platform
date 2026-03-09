use std::collections::VecDeque;

#[derive(Clone)]
pub struct CallbackRecord {
    pub node: usize,
    pub latency: f32,
    pub iteration: usize,
}

pub struct CallbackLog {
    history: VecDeque<CallbackRecord>,
    capacity: usize,
}

impl CallbackLog {
    pub fn new(capacity: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn record(&mut self, rec: CallbackRecord) {
        if self.history.len() == self.capacity {
            self.history.pop_front();
        }
        self.history.push_back(rec);
    }

    pub fn first_iterations(&self, k: usize) -> Vec<CallbackRecord> {
        self.history.iter().take(k).cloned().collect()
    }
}