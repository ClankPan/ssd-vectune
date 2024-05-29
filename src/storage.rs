use anyhow::Result;
use memmap2::{MmapMut, MmapOptions};
use std::{fs::OpenOptions, sync::Arc};

/* types */

pub trait StorageTrait {
    fn read(&self, offset: usize, dst: &mut [u8]);
    fn write(&self, offset: usize, src: &[u8]);
    fn sector_byte_size(&self) -> usize;
}

pub struct Storage {
    mmap_arc: Arc<MmapMut>,
    sector_byte_size: usize,
}

impl Storage {
    pub fn new_with_empty_file(
        path: &str,
        file_byte_size: u64,
        sector_byte_size: usize,
    ) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        file.set_len(file_byte_size)?;
        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(Self {
            mmap_arc: Arc::new(mmap),
            sector_byte_size,
        })
    }
}

impl StorageTrait for Storage {
    fn read(&self, offset: usize, dst: &mut [u8]) {
        dst.copy_from_slice(&self.mmap_arc[offset..offset + dst.len()])
    }

    fn write(&self, offset: usize, src: &[u8]) {
        let mmap_ref = Arc::clone(&self.mmap_arc);
        unsafe {
            let dest = mmap_ref.as_ptr().add(offset) as *mut u8;
            std::ptr::copy_nonoverlapping(src.as_ptr(), dest, src.len());
        }
    }

    fn sector_byte_size(&self) -> usize {
        self.sector_byte_size
    }
}
