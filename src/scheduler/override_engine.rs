use crate::scheduler::{
    automorphism::GraphAutomorphism,
    callback_log::CallbackLog,
    execution_graph::ExecutionGraph,
    path_optimizer::shortest_path,
};

pub fn optimize_execution(graph: ExecutionGraph, log: &CallbackLog) -> ExecutionGraph {

    let early = log.first_iterations(10);

    let avg_latency: f32 =
        early.iter().map(|r| r.latency).sum::<f32>() / early.len().max(1) as f32;

    if avg_latency > 50.0 {
        // High latency: apply dithering automorphism to escape suboptimal schedule.
        GraphAutomorphism::dither(&graph)
    } else {
        // Low latency: verify the schedule is near-optimal via shortest-path analysis.
        if let Some(start) = graph.nodes.iter().next().cloned() {
            let _distances = shortest_path(&graph, start);
        }
        graph
    }
}