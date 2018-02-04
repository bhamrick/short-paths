#![cfg(test)]

use petgraph::Graph;

use {shortest_path_tree, heap_graph};

#[test]
fn test_shortest_path_tree() {
    // Example graph from http://citeseer.ist.psu.edu/viewdoc/download;jsessionid=7BB56B2ABC7C9113C121413A62AF3974?doi=10.1.1.30.3705&rep=rep1&type=pdf
    // See figure 1a
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

    let spt = shortest_path_tree(&graph, v34, |e| { *e.weight() });

    assert_eq!(spt[&v11].0, 55);
    assert_eq!(spt[&v12].0, 56);
    assert_eq!(spt[&v13].0, 36);
    assert_eq!(spt[&v14].0, 22);
    assert_eq!(spt[&v21].0, 42);
    assert_eq!(spt[&v22].0, 33);
    assert_eq!(spt[&v23].0, 23);
    assert_eq!(spt[&v24].0, 7);
    assert_eq!(spt[&v31].0, 37);
    assert_eq!(spt[&v32].0, 19);
    assert_eq!(spt[&v33].0, 11);
    assert_eq!(spt[&v34].0, 0);

    for (_, &(cost, outgoing_edge)) in spt.iter() {
        if let Some(edge_id) = outgoing_edge {
            let (_, successor) = graph.edge_endpoints(edge_id).unwrap();
            let edge_weight = &graph[edge_id];
            assert_eq!(cost, spt[&successor].0 + edge_weight);
        }
    }
}

#[test]
fn test_heap_graph() {
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

    let spt = shortest_path_tree(&graph, v34, |e| { *e.weight() });

    let (heap_graph, heap_roots) = heap_graph(&graph, v34, &spt, |e| { e });

    println!("{:?}", heap_graph);
    println!("{:?}", heap_roots);
    panic!();
}
