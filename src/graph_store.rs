use crate::storage::StorageTrait;
use crate::utils;

use anyhow::Result;

type StoreIndex = usize;
type SectorIndex = usize;

pub struct GraphStore<S: StorageTrait> {
    storage: S,

    num_vectors: usize,
    vector_dim: usize,
    edge_max_digree: usize,
    num_node_in_sector: usize,
    num_sectors: usize,
    node_byte_size: usize,
}

impl<S> GraphStore<S>
where
    S: StorageTrait,
{
    pub fn new(num_vectors: usize, vector_dim: usize, edge_max_digree: usize, storage: S) -> Self {
        let node_byte_size = vector_dim * 4 + edge_max_digree * 4 + 4;
        let sector_byte_size = storage.sector_byte_size();
        let num_sectors = sector_byte_size / node_byte_size;
        let num_node_in_sector: usize = storage.sector_byte_size() / node_byte_size;
        Self {
            storage,
            num_vectors,
            vector_dim,
            edge_max_digree,
            num_node_in_sector,
            num_sectors,
            node_byte_size,
        }
    }

    fn offset_from_sotre_index(&self, store_index: &StoreIndex) -> usize {
        store_index * self.node_byte_size
    }

    fn offset_from_sector_index(&self, sector_index: &SectorIndex) -> usize {
        sector_index * self.storage.sector_byte_size()
    }

    pub fn read_serialized_node(&self, store_index: &StoreIndex) -> Vec<u8> {
        let offset = self.offset_from_sotre_index(store_index);
        let mut bytes: Vec<u8> = vec![0; self.node_byte_size];
        self.storage.read(offset, &mut bytes);
        bytes
    }

    pub fn read_node(&self, store_index: &StoreIndex) -> Result<(Vec<f32>, Vec<u32>)> {
        let bytes = self.read_serialized_node(store_index);

        let (vector, edges) =
            utils::deserialize_node(&bytes, self.vector_dim, self.edge_max_digree);

        Ok((vector.to_vec(), edges.to_vec()))
    }

    pub fn read_edges(&self, store_index: &StoreIndex) -> Result<Vec<u32>> {
        let (_, edges) = self.read_node(store_index)?;
        Ok(edges)
    }

    pub fn write_node(
        &self,
        store_index: &StoreIndex,
        vector: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()> {
        let bytes = utils::serialize_node(vector, edges);
        let offset = self.offset_from_sector_index(store_index);
        self.storage.write(offset, &bytes);
        Ok(())
    }

    pub fn write_serialized_sector(&self, sector_index: &StoreIndex, bytes: &[u8]) -> Result<()> {
        let offset = self.offset_from_sector_index(sector_index);
        self.storage.write(offset, bytes);
        Ok(())
    }

    // fn store_index_to_sector_index_and_offset(
    //     &self,
    //     store_index: &StoreIndex,
    // ) -> (SectorIndex, usize) {
    //     // アライメントがずれている場合、
    //     let sector_index = *store_index / self.num_node_in_sector;
    //     let offset = (*store_index as usize % self.num_node_in_sector) * self.node_byte_size;
    //     (sector_index, offset)
    // }

    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }
    pub fn num_node_in_sector(&self) -> usize {
        self.num_node_in_sector
    }
    pub fn num_sectors(&self) -> usize {
        self.num_sectors
    }
    pub fn vector_dim(&self) -> usize {
        self.vector_dim
    }
    pub fn edge_max_digree(&self) -> usize {
        self.edge_max_digree
    }
    pub fn node_byte_size(&self) -> usize {
        self.node_byte_size
    }
}

pub struct EdgesIterator<'a, S: StorageTrait> {
    graph: &'a GraphStore<S>,
    index: StoreIndex,
    current_position: usize,
    edges: Vec<u32>,
}

impl<'a, S: StorageTrait> EdgesIterator<'a, S> {
    pub fn new(graph: &'a GraphStore<S>) -> Self {
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

impl<'a, S: StorageTrait> Iterator for EdgesIterator<'a, S> {
    type Item = std::result::Result<(u32, u32), std::io::Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_position >= self.edges.len() {
            // If the current position exceeds the length of the edge list, the next edge set is read
            self.current_position = 0;
            self.index += 1;

            if self.graph.num_vectors() >= self.index {
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

        let result = (self.edges[self.current_position], (self.index) as u32);
        self.current_position += 1;
        Some(Ok(result))
    }
}
