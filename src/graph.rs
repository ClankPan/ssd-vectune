use std::time::SystemTime;

const DIM: usize = 384;
const DIGREE: usize = 70;

pub type Id = u32;
pub type SectorIndex = usize;
pub type StoreIndex = u32;
pub type CacheIndex = usize;
pub type SerializedVector = [u8; VECTOR_SIZE];
pub type SerializedEdges = [u8; EDGES_SIZE];

pub type Vector = [f32; DIM];
pub type Edges = [Id; DIGREE];

pub const PAGE_SIZE: u64 = 64 * 1024; // 1 page is 64KiB
pub const SECTOR_SIZE: usize = 1024 * 1024; // 1 MiB
pub const VECTOR_SIZE: usize = DIM * std::mem::size_of::<f32>();
pub const EDGES_SIZE: usize = DIGREE * std::mem::size_of::<u32>();
pub const NODE_SIZE: usize = VECTOR_SIZE + EDGES_SIZE;
pub const NUM_NODE_IN_SECTOR: usize = SECTOR_SIZE / NODE_SIZE;

pub fn serialize_node<'a>(
    vector: &'a Vector,
    edges: &'a Edges,
) -> (&'a SerializedVector, &'a SerializedEdges) {
    let serialize_vector: &SerializedVector = serialize_vector(vector);
    let serialize_edges: &SerializedEdges = serialize_edges(edges);
    (serialize_vector, serialize_edges)
}

pub fn serialize_vector(vector: &Vector) -> &SerializedVector {
    let serialize_vector: &SerializedVector = bytemuck::cast_slice(vector)
        .try_into()
        .expect("Failed to try into &[u8; DIM*4]");
    serialize_vector
}

pub fn serialize_edges(edges: &Edges) -> &SerializedEdges {
    let serialize_edges: &SerializedEdges = bytemuck::cast_slice(edges)
        .try_into()
        .expect("Failed to try into &[u8; DIGREE*4]");
    serialize_edges
}

pub fn deserialize_node<'a>(
    serialize_vector: &'a SerializedVector,
    serialize_edges: &'a SerializedEdges,
) -> (&'a Vector, &'a Edges) {
    let vector: &Vector = deserialize_vector(serialize_vector);
    let edges: &Edges = deserialize_edges(serialize_edges);

    (vector, edges)
}

pub fn deserialize_vector(serialize_vector: &SerializedVector) -> &Vector {
    let vector: &Vector = bytemuck::try_cast_slice(serialize_vector)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    vector
}

pub fn deserialize_edges(serialize_edges: &SerializedEdges) -> &Edges {
    let edges: &Edges = bytemuck::try_cast_slice(serialize_edges)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    edges
}

// offset is address of the node in a sector buffer
pub fn store_index_to_sector_index_and_offset(store_index: &StoreIndex) -> (SectorIndex, usize) {
    let sector_index = *store_index as usize / NUM_NODE_IN_SECTOR;
    let offset = (*store_index as usize % NUM_NODE_IN_SECTOR) * NODE_SIZE;
    (sector_index, offset)
}

pub trait Storage {
    fn read(&self, offset: usize, dst: &mut [u8]);
    fn write(&mut self, offset: usize, src: &[u8]);
}

pub struct Cache<S: Storage> {
    pub(crate) cache_table: Vec<(CacheIndex, SystemTime)>,
    pub(crate) cache: Vec<[u8; SECTOR_SIZE]>,
    pub(crate) storage: S,
}

impl<S> Cache<S>
where
    S: Storage,
{
    pub fn clear(&mut self) {
        self.cache_table = vec![(usize::MAX, SystemTime::now()); self.cache_table.len()];
        self.cache = vec![[0; SECTOR_SIZE]; self.cache.len()];
    }

    pub fn read_node(&mut self, store_index: &StoreIndex) -> (&Vector, &Edges) {
        let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(store_index);
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

        let serialized_point: &SerializedVector = sector
            [offset_in_sector..offset_in_sector + VECTOR_SIZE]
            .try_into()
            .expect("cannot convert to  &SerializedPoint");
        let serialized_edges: &SerializedEdges = sector
            [offset_in_sector + VECTOR_SIZE..offset_in_sector + VECTOR_SIZE + EDGES_SIZE]
            .try_into()
            .expect("cannot convert to  &SerializedEdges");
        let (point, edges) = deserialize_node(serialized_point, serialized_edges);
        (point, edges)
    }

    pub fn write_edges(&mut self, store_index: &StoreIndex, edges: &Edges) {
        let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(store_index);
        self.cache_table[sector_index].0 = usize::MAX; // This cache is no longer used
        let serialized_edges = serialize_edges(edges);
        let offset = sector_index * SECTOR_SIZE + offset_in_sector + VECTOR_SIZE; // First address of the edges
        self.storage.write(offset, serialized_edges);

        // wip: dirty_sector_table.insert(sector_index);
    }
}
