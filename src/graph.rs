use anyhow::Result;
use std::fs::File;
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use vectune::{GraphInterface, PointInterface};

use crate::graph_store::GraphStore;
use crate::point::Point;
use crate::storage::StorageTrait;

#[derive(Clone)]
pub struct SectoredGraph<S: StorageTrait> {
    size_l: usize,
    size_r: usize,
    size_a: f32,

    graph_store: GraphStore<S>,
    node_index_to_store_index: Vec<u32>,
    start_node_index: u32,
}

impl<S: StorageTrait> SectoredGraph<S> {
    pub fn new(
        graph_store: GraphStore<S>,
        node_index_to_store_index: Vec<u32>,
        start_node_index: u32,
    ) -> Self {
        Self {
            size_l: 125,
            size_r: graph_store.max_edge_degrees(),
            size_a: 2.0,

            graph_store,
            node_index_to_store_index,
            start_node_index,
        }
    }

    pub fn set_start_node_index(&mut self, index: u32) {
        self.start_node_index = index;
    }

    pub fn set_size_l(&mut self, size_l: usize) {
        self.size_l = size_l;
    }
}

impl<S: StorageTrait> GraphInterface<Point> for SectoredGraph<S> {
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
        let (vector, edges) = self.graph_store.read_node(&store_index).unwrap();

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

#[derive(Clone)]
pub struct Graph<S: StorageTrait> {
    size_l: usize,
    size_r: usize,
    size_a: f32,

    graph_store: GraphStore<S>,
    start_node_index: u32,
}

impl<S: StorageTrait> Graph<S> {
    pub fn new(graph_store: GraphStore<S>) -> Self {
        Self {
            size_l: 125,
            size_r: graph_store.max_edge_degrees(),
            size_a: 2.0,
            start_node_index: graph_store.start_id() as u32,
            graph_store,
        }
    }

    pub fn set_start_node_index(&mut self, index: u32) {
        self.start_node_index = index;
    }

    pub fn set_size_l(&mut self, size_l: usize) {
        self.size_l = size_l;
    }
}

impl<S: StorageTrait, P: PointInterface> GraphInterface<P> for Graph<S> {
    fn alloc(&mut self, _point: P) -> u32 {
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

    fn get(&mut self, node_index: &u32) -> (P, Vec<u32>) {
        let store_index = node_index;
        let (vector, edges) = self.graph_store.read_node(&store_index).unwrap();

        (P::from_f32_vec(vector), edges)
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

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphMetadata {
    pub node_index_to_store_index: Vec<u32>,
    pub medoid_node_index: u32,
    pub sector_byte_size: usize,
    pub num_vectors: usize,
    pub vector_dim: usize,
    pub edge_degrees: usize,
}

impl GraphMetadata {
    pub fn new(
        node_index_to_store_index: Vec<u32>,
        medoid_node_index: u32,
        sector_byte_size: usize,
        num_vectors: usize,
        vector_dim: usize,
        edge_degrees: usize,
    ) -> Self {
        Self {
            node_index_to_store_index,
            medoid_node_index,
            sector_byte_size,
            num_vectors,
            vector_dim,
            edge_degrees,
        }
    }

    pub fn load(path: &str) -> Result<Self> {
        let mut file = File::open(path).expect("file not found");

        // ファイルの内容を読み込む
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        // JSONデータをPerson構造体にデシリアライズ
        let metadata: Self = serde_json::from_str(&contents)?;
        Ok(metadata)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let json_string = serde_json::to_string(self)?;
        let mut file = File::create(path)?;
        file.write_all(json_string.as_bytes())?;
        Ok(())
    }

    pub fn load_debug(path: &str, sector_byte_size: usize) -> Result<Self> {
        let mut file = File::open(path).expect("file not found");

        // ファイルの内容を読み込む
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("something went wrong reading the file");

        // JSONデータをPerson構造体にデシリアライズ
        let (node_index_to_store_index, medoid_node_index): (Vec<u32>, u32) =
            serde_json::from_str(&contents)?;
        let metadata = Self {
            node_index_to_store_index,
            medoid_node_index,
            sector_byte_size,
            num_vectors: 100 * 1000000,
            vector_dim: 96,
            edge_degrees: 70 * 2,
        };

        metadata.save(&format!("{path}.debug"))?;

        Ok(metadata)
    }
}
