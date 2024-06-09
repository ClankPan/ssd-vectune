use std::{cell::RefCell, time::SystemTime};

use crate::storage::StorageTrait;

pub type CacheIndex = usize;
pub struct Cache<S: StorageTrait> {
    sector_byte_size: usize,
    node_byte_size: usize,
    storage: S,
    cache_table: RefCell<Vec<(CacheIndex, SystemTime)>>, // Should be therad local
    cache: RefCell<Vec<Vec<u8>>>,
}

impl<S: StorageTrait> Cache<S> {
    pub fn new(num_sector: usize, vector_dim: usize, edge_max_digree: usize, storage: S) -> Self {
        let sector_byte_size: usize = storage.sector_byte_size();
        let node_byte_size: usize = vector_dim * 4 + edge_max_digree * 4 + 4;
        // let num_node_in_sector: usize = sector_byte_size / node_byte_size;
        Self {
            sector_byte_size,
            node_byte_size,
            storage,
            cache_table: RefCell::new(vec![(usize::MAX, SystemTime::now()); num_sector]),
            cache: RefCell::new(vec![vec![]; num_sector]),
        }
    }
}

impl<S> StorageTrait for Cache<S>
where
    S: StorageTrait,
{
    fn read(&self, offset: usize, dst: &mut [u8]) {
        let sector_index = offset / self.sector_byte_size;
        let offset_in_sector = offset % self.sector_byte_size;

        let mut cache_table = self.cache_table.borrow_mut();
        let mut cache = self.cache.borrow_mut();

        let sector = if cache_table[sector_index].0 == usize::MAX {
            // Eviction
            let cache_index = cache_table
                .iter()
                .enumerate()
                .min_by_key(|&(_index, (_, count))| count)
                .unwrap()
                .0;
            // Write data to cache
            let mut buffer = vec![0; self.sector_byte_size];
            self.storage
                .read(sector_index * self.sector_byte_size, &mut buffer);
            // Update cache table
            cache_table[sector_index].0 = cache_index;
            cache[cache_index] = buffer;
            &cache[cache_index]
        } else {
            let cache_index = cache_table[sector_index].0;
            &cache[cache_index]
        };
        // Set the time of the cache used
        cache_table[sector_index].1 = SystemTime::now();

        dst.copy_from_slice(&sector[offset_in_sector..offset_in_sector + self.node_byte_size]);
    }

    fn write(&self, _offset: usize, _src: &[u8]) {
        todo!()
    }

    fn sector_byte_size(&self) -> usize {
        self.sector_byte_size
    }
}
