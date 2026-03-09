use std::collections::HashSet;

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub usize);

#[derive(Clone, Debug)]
pub struct Edge {
    pub from: NodeId,
    pub to: NodeId,
    pub cost: f32,
}

#[derive(Clone)]
pub struct ExecutionGraph {
    pub nodes: HashSet<NodeId>,
    pub edges: Vec<Edge>,
}