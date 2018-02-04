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

#[test]
fn test_2_cycle() {
    let mut graph = Graph::<&str, i32>::new();
    let s = graph.add_node("start");
    let t = graph.add_node("end");
    graph.extend_with_edges(&[
        (s, t, 1),
        (t, s, 1),
    ]);

    let mut path_count = 0;
    let mut prev_length = -1;
    for path in ShortPathsIterator::new(&graph, s, t, |e| { e }).take(100) {
        let mut length = 0;
        for &e in path.iter() {
            length += graph.edge_weight(e).unwrap();
        }
        assert_eq!(length, prev_length + 2);
        prev_length = length;
        path_count += 1;
    }

    assert_eq!(path_count, 100);
}

#[test]
fn test_2x2_grid() {
    let mut graph = Graph::<&str, i32>::new();
    let v11 = graph.add_node("v11");
    let v12 = graph.add_node("v12");
    let v21 = graph.add_node("v21");
    let v22 = graph.add_node("v22");
    graph.extend_with_edges([
        (v11, v12, 1),
        (v12, v11, 1),

        (v11, v21, 1),
        (v21, v11, 1),

        (v12, v22, 1),
        (v22, v12, 1),
        
        (v21, v22, 1),
        (v22, v21, 1),
    ].iter());

    let mut path_count = 0;
    let mut prev_length = 0;
    for path in ShortPathsIterator::new(&graph, v11, v21, |e| { e }).take(100) {
        let mut length = 0;
        for &e in path.iter() {
            length += graph.edge_weight(e).unwrap();
        }
        assert!(length >= prev_length);
        prev_length = length;
        path_count += 1;
    }

    assert_eq!(path_count, 100);
}

#[test]
fn test_4x4_grid() {
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
    let v41 = graph.add_node("v41");
    let v42 = graph.add_node("v42");
    let v43 = graph.add_node("v43");
    let v44 = graph.add_node("v44");
    graph.extend_with_edges([
        (v11, v12, 1),
        (v12, v11, 1),
        (v12, v13, 1),
        (v13, v12, 1),
        (v13, v14, 1),
        (v14, v13, 1),
        (v11, v21, 1),
        (v21, v11, 1),
        (v12, v22, 1),
        (v22, v12, 1),
        (v13, v23, 1),
        (v23, v13, 1),
        (v14, v24, 1),
        (v24, v14, 1),

        (v21, v22, 1),
        (v22, v21, 1),
        (v22, v23, 1),
        (v23, v22, 1),
        (v23, v24, 1),
        (v24, v23, 1),
        (v21, v31, 1),
        (v31, v21, 1),
        (v22, v32, 1),
        (v32, v22, 1),
        (v23, v33, 1),
        (v33, v23, 1),
        (v24, v34, 1),
        (v34, v24, 1),

        (v31, v32, 1),
        (v32, v31, 1),
        (v32, v33, 1),
        (v33, v32, 1),
        (v33, v34, 1),
        (v34, v33, 1),
        (v31, v41, 1),
        (v41, v31, 1),
        (v32, v42, 1),
        (v42, v32, 1),
        (v33, v43, 1),
        (v43, v33, 1),
        (v34, v44, 1),
        (v44, v34, 1),

        (v41, v42, 1),
        (v42, v41, 1),
        (v42, v43, 1),
        (v43, v42, 1),
        (v43, v44, 1),
        (v44, v43, 1),
    ].iter());

    let mut path_count = 0;
    let mut prev_length = 0;
    for path in ShortPathsIterator::new(&graph, v22, v33, |e| { e }).take(100) {
        let mut length = 0;
        for &e in path.iter() {
            length += graph.edge_weight(e).unwrap();
        }
        assert!(length >= prev_length);
        prev_length = length;
        path_count += 1;
    }

    assert_eq!(path_count, 100);
}
