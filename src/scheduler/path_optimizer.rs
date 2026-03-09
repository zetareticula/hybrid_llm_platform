use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use crate::scheduler::execution_graph::{ExecutionGraph, NodeId};

/// BinaryHeap wrapper that orders by cost ascending (min-heap via reversed Ord).
#[derive(PartialEq)]
struct State {
    cost: f32,
    node: NodeId,
}

impl Eq for State {}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) behaves as min-heap for Dijkstra.
        other.cost.total_cmp(&self.cost)
            .then_with(|| self.node.cmp(&other.node))
    }
}

pub fn shortest_path(graph: &ExecutionGraph, start: NodeId) -> HashMap<NodeId, f32> {
    let mut dist: HashMap<NodeId, f32> = HashMap::new();
    let mut heap = BinaryHeap::new();

    dist.insert(start.clone(), 0.0);
    heap.push(State { cost: 0.0, node: start });

    while let Some(State { cost, node }) = heap.pop() {
        for edge in graph.edges.iter().filter(|e| e.from == node) {
            let next      = edge.to.clone();
            let next_cost = cost + edge.cost;

            if dist.get(&next).map_or(true, |&c| next_cost < c) {
                dist.insert(next.clone(), next_cost);
                heap.push(State { cost: next_cost, node: next });
            }
        }
    }

    dist
}