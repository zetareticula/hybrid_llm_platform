use rand::seq::SliceRandom;
use rand::thread_rng;

use crate::scheduler::execution_graph::{ExecutionGraph, NodeId};

pub struct GraphAutomorphism;

impl GraphAutomorphism {

    pub fn dither(graph: &ExecutionGraph) -> ExecutionGraph {
        let mut rng = thread_rng();

        let mut permuted_nodes: Vec<NodeId> = graph.nodes.iter().cloned().collect();
        permuted_nodes.shuffle(&mut rng);

        // apply symmetry preserving permutation
        let mut new_edges = Vec::new();

        for e in &graph.edges {
            new_edges.push(e.clone());
        }

        ExecutionGraph {
            nodes: permuted_nodes.into_iter().collect(),
            edges: new_edges,
        }
    }
}