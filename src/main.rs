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
    let serialize_point: &SerializedPoint = bytemuck::cast_slice(point)
        .try_into()
        .expect("Failed to try into &[u8; DIM*4]");
    let serialize_edges: &SerializedEdges = bytemuck::cast_slice(edges)
        .try_into()
        .expect("Failed to try into &[u8; DIGREE*4]");
    (serialize_point, serialize_edges)
}

fn deserialize_node<'a>(
    serialize_point: &'a SerializedPoint,
    serialize_edges: &'a SerializedEdges,
) -> (&'a Point, &'a Edges) {
    let point: &Point = bytemuck::try_cast_slice(serialize_point)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    let edges: &Edges = bytemuck::try_cast_slice(serialize_edges)
        .expect("Failed to deserialize embeddings")
        .try_into()
        .expect("Failed to try into &[f32; DIM]");

    (point, edges)
}

// offset is address of the node in a sector buffer
fn store_index_to_sector_index_and_offset(store_index: &StoreIndex) -> (SectorIndex, usize) {
    let sector_index = *store_index as usize / NUM_NODE_IN_SECTOR;
    let offset = (*store_index as usize % NUM_NODE_IN_SECTOR) * NODE_SIZE;
    (sector_index, offset)
}


trait Storage {
    fn read(&self, offset: u64, dst: &mut [u8]);
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
    fn read(&mut self, store_index: &StoreIndex) -> (&Point, &Edges) {
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
            self.storage.read((sector_index * SECTOR_SIZE) as u64, buffer);
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
}
