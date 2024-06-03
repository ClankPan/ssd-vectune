use std::time::SystemTime;

use crate::storage::StorageTrait;

pub type CacheIndex = usize;
pub struct Cache<S: StorageTrait> {
    sector_byte_size: usize,
    storage: S,
    cache_table: Vec<(CacheIndex, SystemTime)>,
    cache: Vec<Vec<u8>>,
}

impl<S> StorageTrait for Cache<S>
where
    S: StorageTrait,
{
    fn read(&self, offset: usize, dst: &mut [u8]) {
        todo!()
    }

    fn write(&self, offset: usize, src: &[u8]) {
        todo!()
    }

    fn sector_byte_size(&self) -> usize {
        self.sector_byte_size
    }
}















use std::{fs::OpenOptions, time::SystemTime};

use anyhow::Result;
use memmap2::{MmapMut, MmapOptions};

use crate::utils;

// const DIM: usize = 384;
// const DIGREE: usize = 70;

pub type Id = u32;
pub type SectorIndex = usize;
pub type StoreIndex = u32;
pub type CacheIndex = usize;
pub type SerializedVector = [u8];
pub type SerializedEdges = [u8];

pub type Vector = [f32];
pub type Edges = [Id];

// pub const PAGE_SIZE: u64 = 64 * 1024; // 1 page is 64KiB
pub const SECTOR_SIZE: usize = 1024 * 1024; // 1 MiB
                                            // pub const VECTOR_SIZE: usize = DIM * std::mem::size_of::<f32>();
                                            // pub const EDGES_SIZE: usize = DIGREE * std::mem::size_of::<u32>();
                                            // pub const NODE_SIZE: usize = VECTOR_SIZE + EDGES_SIZE;
                                            // pub const NUM_NODE_IN_SECTOR: usize = SECTOR_SIZE / NODE_SIZE;

pub trait StorageTrait {
    // fn new_with_empty_file(
    //     path: &str,
    //     file_byte_size: usize,
    // ) -> Result<Self>;
    fn read(&self, offset: usize, dst: &mut [u8]);
    fn write(&mut self, offset: usize, src: &[u8]);
}

pub struct Storage {
    mmap: MmapMut,
}

impl Storage {
    pub fn new_with_empty_file(path: &str, file_byte_size: u64) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(file_byte_size)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self { mmap })
    }
}

impl StorageTrait for Storage {
    fn read(&self, offset: usize, dst: &mut [u8]) {
        dst.copy_from_slice(&self.mmap[offset..offset + dst.len()])
    }

    fn write(&mut self, offset: usize, src: &[u8]) {
        self.mmap[offset..offset + src.len()].copy_from_slice(src)
    }
}

pub struct Cache<S: StorageTrait> {
    pub(crate) cache_table: Vec<(CacheIndex, SystemTime)>,
    pub(crate) cache: Vec<[u8; SECTOR_SIZE]>,
    pub(crate) storage: S,

    vector_dim: usize,
    edge_max_digree: usize,
    num_node_in_sector: usize,
    node_byte_size: usize,
}

impl<S> Cache<S>
where
    S: StorageTrait,
{
    pub fn new(num_sector: usize, vector_dim: usize, edge_max_digree: usize, storage: S) -> Self {
        let node_byte_size = vector_dim * 4 + edge_max_digree * 4 + 4;
        let num_node_in_sector: usize = SECTOR_SIZE / node_byte_size;
        Self {
            cache_table: vec![(usize::MAX, SystemTime::now()); num_sector],
            cache: vec![[0; SECTOR_SIZE]; num_sector],
            storage,
            vector_dim,
            edge_max_digree,
            num_node_in_sector,
            node_byte_size,
        }
    }

    // offset is address of the node in a sector buffer
    pub fn store_index_to_sector_index_and_offset(
        &self,
        store_index: &StoreIndex,
    ) -> (SectorIndex, usize) {
        let sector_index = *store_index as usize / self.num_node_in_sector;
        let offset = (*store_index as usize % self.num_node_in_sector) * self.node_byte_size;
        (sector_index, offset)
    }

    pub fn clear(&mut self) {
        self.cache_table = vec![(usize::MAX, SystemTime::now()); self.cache_table.len()];
        self.cache = vec![[0; SECTOR_SIZE]; self.cache.len()];
    }

    pub fn read_node(&mut self, store_index: &StoreIndex) -> (&Vector, &Edges) {
        let (sector_index, offset_in_sector) =
            self.store_index_to_sector_index_and_offset(store_index);
        let sector = if self.cache_table[sector_index].0 == usize::MAX {
            // Eviction
            let cache_index = self
                .cache_table
                .iter()
                .enumerate()
                .min_by_key(|&(_index, (_, count))| count)
                .unwrap()
                .0;
            // Write data to cache
            let buffer = &mut self.cache[cache_index];
            self.storage.read(sector_index * SECTOR_SIZE, buffer);
            // Update cache table
            self.cache_table[sector_index].0 = cache_index;
            &self.cache[cache_index]
        } else {
            let cache_index = self.cache_table[sector_index].0;
            &self.cache[cache_index]
        };
        // Set the time of the cache used
        self.cache_table[sector_index].1 = SystemTime::now();

        let serialized_node = &sector[offset_in_sector..offset_in_sector + self.node_byte_size];
        let (vector, edges) =
            utils::deserialize_node(serialized_node, self.vector_dim, self.edge_max_digree);
        (vector, edges)
    }

    pub fn write_edges(&mut self, store_index: &StoreIndex, edges: &Edges) {
        let (sector_index, offset_in_sector) =
            self.store_index_to_sector_index_and_offset(store_index);
        self.cache_table[sector_index].0 = usize::MAX; // This cache is no longer used
        let (serialized_edges, serialized_edges_len) = utils::serialize_edges(edges);
        let mut combined = Vec::with_capacity(serialized_edges.len() + 4);
        combined.extend_from_slice(&serialized_edges_len);
        combined.extend_from_slice(serialized_edges);
        let offset = sector_index * SECTOR_SIZE + offset_in_sector + self.vector_dim * 4; // First address of the edges
        self.storage.write(offset, &combined);

        // wip: dirty_sector_table.insert(sector_index);
    }

    pub fn write_serialized_sector(&mut self, sector_index: &u32, bytes: &[u8]) {}
}
