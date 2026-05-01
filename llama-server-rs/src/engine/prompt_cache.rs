use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct PromptCacheEntry {
    pub key: u64,
    pub blob: Vec<u8>,
}

#[derive(Debug)]
pub struct PromptCache {
    max_entries: usize,
    map: HashMap<u64, Vec<u8>>,
    lru: VecDeque<u64>,
}

impl PromptCache {
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            map: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    pub fn get(&mut self, key: u64) -> Option<Vec<u8>> {
        let value = self.map.get(&key).cloned();
        if value.is_some() {
            self.touch(key);
        }
        value
    }

    pub fn put(&mut self, key: u64, blob: Vec<u8>) {
        self.map.insert(key, blob);
        self.touch(key);
        while self.map.len() > self.max_entries {
            if let Some(old) = self.lru.pop_back() {
                self.map.remove(&old);
            }
        }
    }

    fn touch(&mut self, key: u64) {
        self.lru.retain(|k| *k != key);
        self.lru.push_front(key);
    }
}
