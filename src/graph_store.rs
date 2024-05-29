use crate::storage::StorageTrait;
use crate::utils;

use anyhow::Result;

type StoreIndex = usize;
type SectorIndex = usize;

pub struct GraphStore<S: StorageTrait> {
    storage: S,

    vector_dim: usize,
    edge_max_digree: usize,
    num_node_in_sector: usize,
    num_sector: usize,
    node_byte_size: usize,
}

impl<S> GraphStore<S>
where
    S: StorageTrait,
{
    pub fn new(num_sector: usize, vector_dim: usize, edge_max_digree: usize, storage: S) -> Self {
        let node_byte_size = vector_dim * 4 + edge_max_digree * 4 + 4;
        let num_node_in_sector: usize = storage.sector_byte_size() / node_byte_size;
        Self {
            storage,
            vector_dim,
            edge_max_digree,
            num_node_in_sector,
            num_sector,
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

        let (vector, edges) = utils::deserialize_node(
            &bytes,
            self.vector_dim as usize,
            self.edge_max_digree as usize,
        );

        Ok((vector.to_vec(), edges.to_vec()))
    }

    pub fn read_edges(&self, store_index: &StoreIndex) -> Result<Vec<u32>> {
        let (_, edges) = self.read_node(store_index)?;
        Ok(edges)
    }

    pub fn write_node(
        &mut self,
        store_index: &StoreIndex,
        vector: &Vec<f32>,
        edges: &Vec<u32>,
    ) -> Result<()> {
        let bytes = utils::serialize_node(vector, edges);
        let offset = self.offset_from_sector_index(store_index);
        Ok(self.storage.write(offset, &bytes))
    }

    pub fn write_serialized_sector(
        &self,
        sector_index: &StoreIndex,
        bytes: &[u8],
    ) -> Result<()> {
        let offset = self.offset_from_sector_index(sector_index);
        Ok(self.storage.write(offset, bytes))
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
