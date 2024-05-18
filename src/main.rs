#[cfg(test)]
mod tests;

use std::time::SystemTime;

fn main() {
    println!("Hello, world!");
}


const DIM: usize = 384;
const DIGREE: usize = 70;

type Id = u32;
type SectorIndex = usize;
type StoreIndex = u32;
type CacheIndex = usize;
type SerializedPoint = [u8; POINT_SIZE];
type SerializedEdges = [u8; EDGES_SIZE];

type Point = [f32; DIM];
type Edges = [Id; DIGREE];

const PAGE_SIZE: u64 = 64 * 1024; // 1 page is 64KiB
const SECTOR_SIZE: usize = 1024 * 1024; // 1 MiB
const POINT_SIZE: usize = DIM * std::mem::size_of::<f32>();
const EDGES_SIZE: usize = DIGREE * std::mem::size_of::<u32>();
const NODE_SIZE: usize = POINT_SIZE + EDGES_SIZE;
const NUM_NODE_IN_SECTOR: usize = SECTOR_SIZE / NODE_SIZE;

fn serialize_node<'a>(
    point: &'a Point,
    edges: &'a Edges,
) -> (&'a SerializedPoint, &'a SerializedEdges) {
    let serialize_point: &SerializedPoint = serialize_point(point);
    let serialize_edges: &SerializedEdges = serialize_edges(edges);
    (serialize_point, serialize_edges)
}

fn serialize_point<'a>(
    point: &'a Point,
) -> &'a SerializedPoint {
    let serialize_point: &SerializedPoint = bytemuck::cast_slice(point)
        .try_into()
        .expect("Failed to try into &[u8; DIM*4]");
    serialize_point
}

fn serialize_edges<'a>(
    edges: &'a Edges,
) -> &'a SerializedEdges {
    let serialize_edges: &SerializedEdges = bytemuck::cast_slice(edges)
        .try_into()
        .expect("Failed to try into &[u8; DIGREE*4]");
    serialize_edges
}

fn deserialize_node<'a>(
    serialize_point: &'a SerializedPoint,
    serialize_edges: &'a SerializedEdges,
) -> (&'a Point, &'a Edges) {
    let point: &Point = deserialize_point(serialize_point);
    let edges: &Edges = deserialize_edges(serialize_edges);

    (point, edges)
}

fn deserialize_point<'a>(
    serialize_point: &'a SerializedPoint
) -> &'a Point {
    let point: &Point = bytemuck::try_cast_slice(serialize_point)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    point
}

fn deserialize_edges<'a>(
    serialize_edges: &'a SerializedEdges,
) -> &'a Edges {
    let edges: &Edges = bytemuck::try_cast_slice(serialize_edges)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    edges
}

// offset is address of the node in a sector buffer
fn store_index_to_sector_index_and_offset(store_index: &StoreIndex) -> (SectorIndex, usize) {
    let sector_index = *store_index as usize / NUM_NODE_IN_SECTOR;
    let offset = (*store_index as usize % NUM_NODE_IN_SECTOR) * NODE_SIZE;
    (sector_index, offset)
}


trait Storage {
    fn read(&self, offset: usize, dst: &mut [u8]);
    fn write(&mut self, offset: usize, src: &[u8]);
}

struct Cache<S: Storage> {
    cache_table: Vec<(CacheIndex, SystemTime)>,
    cache: Vec<[u8; SECTOR_SIZE]>,
    storage: S
}

impl<S> Cache<S>
where
    S: Storage,
{   
    fn clear(&mut self) {
        self.cache_table = vec![(usize::MAX, SystemTime::now()); self.cache_table.len()];
        self.cache =  vec![[0; SECTOR_SIZE]; self.cache.len()];
    }

    fn read_node(&mut self, store_index: &StoreIndex) -> (&Point, &Edges) {
        let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(store_index);
        let sector = if self.cache_table[sector_index].0 == usize::MAX {
            // Eviction
            let cache_index = self.cache_table
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

        let serialized_point: &SerializedPoint = sector[offset_in_sector..offset_in_sector + POINT_SIZE].try_into().expect("cannot convert to  &SerializedPoint");
        let serialized_edges: &SerializedEdges = sector[offset_in_sector + POINT_SIZE..offset_in_sector + POINT_SIZE + EDGES_SIZE].try_into().expect("cannot convert to  &SerializedEdges");
        let (point, edges) = deserialize_node(serialized_point, serialized_edges);
        (point, edges)
    }

    fn write_edges(&mut self, store_index: &StoreIndex, edges: &Edges) {
        let (sector_index, offset_in_sector) = store_index_to_sector_index_and_offset(store_index);
        self.cache_table[sector_index].0 = usize::MAX; // This cache is no longer used
        let serialized_edges = serialize_edges(edges);
        let offset = sector_index * SECTOR_SIZE + offset_in_sector + POINT_SIZE; // First address of the edges
        self.storage.write(offset, serialized_edges);

        // wip: dirty_sector_table.insert(sector_index);
    }
}
