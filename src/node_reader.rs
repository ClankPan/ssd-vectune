use crate::point::Point;
use anyhow::anyhow;
use anyhow::Result;
use memmap2::MmapMut;
use memmap2::MmapOptions;
use std::fs::OpenOptions;
use vectune::PointInterface;

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
        let vector_size = (vector_dim * 4) as usize;
        let edges_size = (edge_digree * 4) as usize;
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

        let vector: Vec<f32> = bytemuck::try_cast_slice(&bytes[..self.vector_size])
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();
        let edges: Vec<u32> = bytemuck::try_cast_slice(&bytes[self.vector_size..])
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();
        Ok((vector, edges))
    }

    fn read_edges(&self, index: &VectorIndex) -> Result<Vec<u32>> {
        let start = self.header_size + index * self.node_size;
        let end = start + self.node_size;
        let bytes = &self.mmap[start..end];

        let edges: Vec<u32> = bytemuck::try_cast_slice(&bytes[self.vector_size..])
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();
        Ok(edges)
    }

    fn write_node(
        &mut self,
        index: &VectorIndex,
        vector: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()> {
        let start = self.header_size + index * self.node_size;
        let end = start + self.node_size;
        // let bytes = &self.mmap[start..end];

        let serialize_vector: &[u8] = bytemuck::cast_slice(vector)
            .try_into()
            .expect("Failed to try into &[u8; DIM*4]");
        let serialize_edges: &[u8] = bytemuck::cast_slice(edges)
            .try_into()
            .expect("Failed to try into &[u8; DIGREE*4]");
        let mut combined = Vec::with_capacity(serialize_vector.len() + serialize_edges.len());
        combined.extend_from_slice(serialize_vector);
        combined.extend_from_slice(serialize_edges);

        self.mmap[start..end].copy_from_slice(&combined);

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
    use rand::{rngs::SmallRng, Rng, SeedableRng};
    use rayon::result;

    use super::{GraphOnStorage, GraphOnStorageTrait};

    const SEED: u64 = 123456;

    #[test]
    fn testing_graph_on_storage() {
        let mut graph_on_stroage =
            GraphOnStorage::new_with_empty_file("test_vectors/unordered_graph", 1000, 96, 70 * 2)
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
