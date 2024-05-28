use crate::point::Point;
use anyhow::Result;
use memmap2::MmapMut;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use vectune::PointInterface;
use crate::utils;


type VectorIndex = usize;

pub trait GraphOnStorageTrait {
    fn read_node(&self, index: &VectorIndex) -> Result<(Vec<f32>, Vec<u32>)>;
    fn read_edges(&self, index: &VectorIndex) -> Result<Vec<u32>>;
    fn write_node(
        &mut self,
        index: &VectorIndex,
        vectors: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()>;
    fn get_num_vectors(&self) -> usize;
    fn get_vector_dim(&self) -> usize;
    fn get_edge_digree(&self) -> usize;
}

pub struct GraphOnStorage {
    mmap: MmapMut,
    num_vectors: u32,
    vector_dim: u32,
    edge_digree: u32,
    header_size: usize,
    vector_size: usize,
    _edges_size: usize,
    node_size: usize,
}

impl GraphOnStorage {
    pub fn new_with_empty_file(
        path: &str,
        num_vectors: u32,
        vector_dim: u32,
        edge_digree: u32,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        let header_size = 12;
        let vector_size = (bytemuck::cast_slice::<f32, u8>(&vec![0.1; vector_dim as usize])).len();
        let edges_size = (bytemuck::cast_slice::<u32, u8>(&vec![1; edge_digree as usize])).len() + 4; // the 4 byte is for Vec length
        let node_size = vector_size + edges_size;

        let file_size = header_size + node_size * num_vectors as usize;
        file.set_len(file_size as u64)?;

        let mut mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        mmap[0..4].copy_from_slice(&num_vectors.to_le_bytes());
        mmap[4..8].copy_from_slice(&vector_dim.to_le_bytes());
        mmap[8..12].copy_from_slice(&edge_digree.to_le_bytes());

        assert_eq!(vector_dim, Point::dim());

        Ok(Self {
            mmap,
            num_vectors,
            vector_dim,
            edge_digree,
            header_size,
            vector_size,
            _edges_size: edges_size,
            node_size,
        })
    }
}

impl GraphOnStorageTrait for GraphOnStorage {
    fn read_node(&self, index: &VectorIndex) -> Result<(Vec<f32>, Vec<u32>)> {
        let start = self.header_size + index * self.node_size;
        let end = start + self.node_size;
        let bytes = &self.mmap[start..end];

        let (vector, edges) = utils::deserialize_node(bytes, self.vector_dim as usize, self.edge_digree as usize);

        Ok((vector.to_vec(), edges.to_vec()))
    }

    fn read_edges(&self, index: &VectorIndex) -> Result<Vec<u32>> {
        let start = self.header_size + index * self.node_size;
        let end = start + self.node_size;
        let bytes = &self.mmap[start..end];

        let (_vector, edges) = utils::deserialize_node(bytes, self.vector_dim as usize, self.edge_digree as usize);
        Ok(edges.to_vec())
    }

    fn write_node(
        &mut self,
        index: &VectorIndex,
        vector: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()> {

        let serialized_node = utils::serialize_node(vector, edges);

        let start = self.header_size + index * self.node_size;
        let end = start + std::cmp::min(self.node_size, serialized_node.len());

        self.mmap[start..end].copy_from_slice(&serialized_node);

        Ok(())
    }

    fn get_num_vectors(&self) -> usize {
        self.num_vectors as usize
    }

    fn get_vector_dim(&self) -> usize {
        self.vector_dim as usize
    }

    fn get_edge_digree(&self) -> usize {
        self.edge_digree as usize
    }
}

#[cfg(test)]
mod tests {
    use super::{GraphOnStorage, GraphOnStorageTrait};
    use rand::{rngs::SmallRng, Rng, SeedableRng};

    const SEED: u64 = 123456;

    #[test]
    fn testing_graph_on_storage() {
        let mut graph_on_stroage = GraphOnStorage::new_with_empty_file(
            "test_vectors/test_unordered_graph",
            1000,
            96,
            70 * 2,
        )
        .unwrap();

        let mut rng = SmallRng::seed_from_u64(SEED);

        let index = 10;
        let random_vector: Vec<f32> = (0..96).map(|_| rng.gen::<f32>()).collect();
        let random_edges: Vec<u32> = (0..70 * 2).map(|_| rng.gen::<u32>()).collect();
        graph_on_stroage
            .write_node(&index, &random_vector, &random_edges)
            .unwrap();
        let (result_vector, result_edges) = graph_on_stroage.read_node(&index).unwrap();
        assert_eq!(random_vector, result_vector);
        assert_eq!(random_edges, result_edges)
    }
}

pub struct EdgesIterator<'a, G: GraphOnStorageTrait> {
    graph: &'a G,
    index: VectorIndex,
    current_position: usize,
    edges: Vec<u32>,
}

impl<'a, G: GraphOnStorageTrait> EdgesIterator<'a, G> {
    pub fn new(graph: &'a G) -> Self {
        let edges = match graph.read_edges(&0) {
            Ok(edges) => edges,
            Err(_) => panic!(), // wip
        };

        EdgesIterator {
            graph,
            index: 0,
            current_position: 0,
            edges,
        }
    }
}

impl<'a, G: GraphOnStorageTrait> Iterator for EdgesIterator<'a, G> {
    type Item = std::result::Result<(u32, u32), std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.edges.len() {
            // If the current position exceeds the length of the edge list, the next edge set is read
            self.current_position = 0;
            self.index += 1;

            if self.graph.get_num_vectors() >= self.index {
                return None;
            } else {
                match self.graph.read_edges(&self.index) {
                    Ok(new_edges) => {
                        if new_edges.is_empty() {
                            return None;
                        }
                        self.edges = new_edges;
                    }
                    Err(e) => {
                        return Some(Err(std::io::Error::new(
                            std::io::ErrorKind::Other,
                            e.to_string(),
                        )))
                    }
                }
            }
        }

        // let result = self.edges.get(self.current_position).map(|&e| Ok((e, self.index as u32)));
        let result = (self.edges[self.current_position], (self.index) as u32);
        self.current_position += 1;
        Some(Ok(result))
    }
}