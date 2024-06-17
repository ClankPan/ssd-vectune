use crate::point::Point;
use anyhow::anyhow;
use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::marker::PhantomData;
use vectune::PointInterface;

type VectorIndex = usize;

pub trait OriginalVectorReaderTrait<T>
where
    T: bytemuck::Pod,
{
    fn read(&self, index: &VectorIndex) -> Result<Vec<T>>;
    fn read_with_range(&self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<T>>>;
    fn get_num_vectors(&self) -> usize;
    fn get_vector_dim(&self) -> usize;
}

pub struct OriginalVectorReader<T> {
    mmap: Mmap,
    num_vectors: usize,
    vector_dim: usize,
    start_offset: usize,
    phantom: PhantomData<T>,
}
impl<T> OriginalVectorReader<T> {
    pub fn new(path: &str) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let num_vectors = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
        let vector_dim = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
        let start_offset = 8;

        // assert_eq!(vector_dim, Point::dim() as usize);

        Ok(Self {
            mmap,
            num_vectors,
            vector_dim,
            start_offset,
            phantom: PhantomData,
        })
    }

    pub fn new_with(path: &str, m: usize) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let num_vectors = m * 1000000; //  million
        let vector_dim = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
        let start_offset = 8;

        assert_eq!(vector_dim, Point::dim() as usize);

        Ok(Self {
            mmap,
            num_vectors,
            vector_dim,
            start_offset,
            phantom: PhantomData,
        })
    }
}

impl<T: bytemuck::Pod> OriginalVectorReaderTrait<T> for OriginalVectorReader<T> {
    fn read(&self, index: &VectorIndex) -> Result<Vec<T>> {
        let start = self.start_offset + index * self.vector_dim * 4;
        let end = start + self.vector_dim * 4;
        let bytes = &self.mmap[start..end];
        let vector: Vec<T> = bytemuck::try_cast_slice(bytes)
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();
        Ok(vector)
    }

    fn read_with_range(&self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<T>>> {
        let start = self.start_offset + start * self.vector_dim * 4;
        let end = self.start_offset + end * self.vector_dim * 4;
        let bytes = &self.mmap[start..end];
        let vectors: Vec<T> = bytemuck::try_cast_slice(bytes)
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();
        let vectors = vectors
            .chunks(self.vector_dim)
            .into_iter()
            .map(|chunk| chunk.to_vec())
            .collect();
        Ok(vectors)
    }

    fn get_num_vectors(&self) -> usize {
        self.num_vectors
    }

    fn get_vector_dim(&self) -> usize {
        self.vector_dim
    }
}
