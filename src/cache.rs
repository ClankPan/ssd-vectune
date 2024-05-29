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
