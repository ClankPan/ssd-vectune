use vectune::GraphInterface;

use crate::point::Point;
use crate::GraphStore;
use crate::Storage;

pub struct Graph {
    size_l: usize,
    size_r: usize,
    size_a: f32,

    graph_store: GraphStore<Storage>,
    node_index_to_store_index: Vec<u32>,
    start_node_index: u32,
}

impl Graph {
    pub fn new(
        graph_store: GraphStore<Storage>,
        node_index_to_store_index: Vec<u32>,
        start_node_index: u32,
    ) -> Self {
        Self {
            size_l: 125,
            size_r: graph_store.edge_max_digree(),
            size_a: 2.0,

            graph_store,
            node_index_to_store_index,
            start_node_index,
        }
    }

    pub fn set_start_node_index(&mut self, index: u32) {
        self.start_node_index = index;
    }
}

impl GraphInterface<Point> for Graph {
    fn alloc(&mut self, _point: Point) -> u32 {
        todo!()
    }

    fn free(&mut self, _id: &u32) {
        todo!()
    }

    fn cemetery(&self) -> Vec<u32> {
        vec![]
    }

    fn clear_cemetery(&mut self) {
        todo!()
    }

    fn backlink(&self, _id: &u32) -> Vec<u32> {
        todo!()
    }

    fn get(&mut self, node_index: &u32) -> (Point, Vec<u32>) {
        let store_index = self.node_index_to_store_index[*node_index as usize];
        let (vector, edges) = self.graph_store.read_node(&(store_index as usize)).unwrap();

        (Point::from_f32_vec(vector), edges)
    }

    fn size_l(&self) -> usize {
        self.size_l
    }

    fn size_r(&self) -> usize {
        self.size_r
    }

    fn size_a(&self) -> f32 {
        self.size_a
    }

    fn start_id(&self) -> u32 {
        self.start_node_index
    }

    fn overwirte_out_edges(&mut self, _id: &u32, _edges: Vec<u32>) {
        todo!()
    }
}
