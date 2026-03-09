use pyo3::prelude::*;
use ndarray::Array2;

/// Quantum-assisted token reordering
pub fn reorder_tokens_quantum(tokens: &Vec<String>, embeddings: Array2<f32>) -> Vec<String> {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let module = PyModule::from_code(
        py,
        include_str!("../../python_quantum/quantum_optimizer.py"),
        "quantum_optimizer.py",
        "",
    ).unwrap();

    let emb_vec: Vec<Vec<f32>> = embeddings.outer_iter().map(|row| row.to_vec()).collect();

    let ordering: Vec<usize> = module
        .getattr("quantum_optimize_tokens").unwrap()
        .call1((emb_vec,))
        .unwrap()
        .extract()
        .unwrap();

    // Reorder tokens
    ordering.iter().map(|&i| tokens[i].clone()).collect()
}