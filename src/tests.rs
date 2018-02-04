#![cfg(test)]

use petgraph::Graph;

use ShortPathsIterator;

#[test]
fn test_short_paths() {
    // Example graph from the source paper (Figure 1a)
    let mut graph = Graph::<&str, i32>::new();
    let v11 = graph.add_node("v11");
    let v12 = graph.add_node("v12");
    let v13 = graph.add_node("v13");
    let v14 = graph.add_node("v14");
    let v21 = graph.add_node("v21");
    let v22 = graph.add_node("v22");
    let v23 = graph.add_node("v23");
    let v24 = graph.add_node("v24");
    let v31 = graph.add_node("v31");
    let v32 = graph.add_node("v32");
    let v33 = graph.add_node("v33");
    let v34 = graph.add_node("v34");
    graph.extend_with_edges(&[
        (v11, v12, 2),
        (v12, v13, 20),
        (v13, v14, 14),
        (v11, v21, 13),
        (v12, v22, 27),
        (v13, v23, 14),
        (v14, v24, 15),
        (v21, v22, 9),
        (v22, v23, 10),
        (v23, v24, 25),
        (v21, v31, 15),
        (v22, v32, 20),
        (v23, v33, 12),
        (v24, v34, 7),
        (v31, v32, 18),
        (v32, v33, 8),
        (v33, v34, 11),
    ]);

    let mut path_count = 0;
    let mut prev_length = 0;
    for path in ShortPathsIterator::new(&graph, v11, v34, |e| { e }) {
        let mut length = 0;
        for &e in path.iter() {
            length += graph.edge_weight(e).unwrap();
        }
        assert!(length >= prev_length);
        prev_length = length;
        path_count += 1;
    }

    assert_eq!(path_count, 10);
}
