// use crate::point::Point;
use anyhow::anyhow;
use anyhow::Result;
use memmap2::Mmap;
use std::fs::File;
use std::io::BufReader;
use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::marker::PhantomData;
// use vectune::PointInterface;

type VectorIndex = usize;

pub trait OriginalVectorReaderTrait<T>
where
    T: bytemuck::Pod,
{
    fn read(&self, index: &VectorIndex) -> Result<Vec<T>>;
    fn read_with_range(&mut self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<T>>>;
    fn get_num_vectors(&self) -> usize;
    fn get_vector_dim(&self) -> usize;
}

pub struct OriginalVectorReader<T> {
    mmap: Mmap,
    num_vectors: usize,
    vector_dim: usize,
    start_offset: usize,
    buf_reader: BufReader<File>,
    phantom: PhantomData<T>,
}
impl<T> OriginalVectorReader<T> {
    pub fn new(file_path: &str) -> Result<Self> {
        let file = File::open(file_path)?;
        let _file_path = file_path.to_string();
        let mmap = unsafe { Mmap::map(&file)? };
        let num_vectors = u32::from_le_bytes(mmap[0..4].try_into()?) as usize;
        let vector_dim = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
        let start_offset = 8;
        let buf_reader: BufReader<File> = BufReader::new(file);

        // assert_eq!(vector_dim, Point::dim() as usize);

        Ok(Self {
            mmap,
            num_vectors,
            vector_dim,
            start_offset,
            buf_reader,
            phantom: PhantomData,
        })
    }

    pub fn new_with(file_path: &str, m: usize) -> Result<Self> {
        let file = File::open(file_path)?;
        let _file_path = file_path.to_string();
        let mmap = unsafe { Mmap::map(&file)? };
        let num_vectors = m * 1000000; //  million
        let vector_dim = u32::from_le_bytes(mmap[4..8].try_into()?) as usize;
        let start_offset = 8;
        let buf_reader: BufReader<File> = BufReader::new(file);

        // assert_eq!(vector_dim, Point::dim() as usize);

        Ok(Self {
            mmap,
            num_vectors,
            vector_dim,
            start_offset,
            buf_reader,
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

    fn read_with_range(&mut self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<T>>> {
        // startとendのバイト位置を計算
        let start_pos = self.start_offset + start * self.vector_dim * 4;
        let end_pos = self.start_offset + end * self.vector_dim * 4;
        let length = end_pos - start_pos;

        // ファイルポインタをstart_posに移動
        self.buf_reader.seek(SeekFrom::Start(start_pos as u64))?;

        // 必要なバイト数を読み込む
        let mut buffer = vec![0u8; length];
        self.buf_reader.read_exact(&mut buffer)?;

        // バイトをT型のベクトルに変換
        let vectors: Vec<T> = bytemuck::try_cast_slice(&buffer)
            .map_err(|e| anyhow!("PodCastError: {:?}", e))?
            .to_vec();

        // ベクトルをself.vector_dimごとに区切る
        let vectors = vectors
            .chunks(self.vector_dim)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok(vectors)
    }

    // fn read_with_range(&self, start: &VectorIndex, end: &VectorIndex) -> Result<Vec<Vec<T>>> {
    //     let start = self.start_offset + start * self.vector_dim * 4;
    //     let end = self.start_offset + end * self.vector_dim * 4;
    //     let bytes: &[u8] = &self.mmap[start..end];
    //     let vectors: Vec<T> = bytemuck::try_cast_slice(bytes)
    //         .map_err(|e| anyhow!("PodCastError: {:?}", e))?
    //         .to_vec();
    //     let vectors = vectors
    //         .chunks(self.vector_dim)
    //         .into_iter()
    //         .map(|chunk| chunk.to_vec())
    //         .collect();
    //     Ok(vectors)
    // }

    fn get_num_vectors(&self) -> usize {
        self.num_vectors
    }

    fn get_vector_dim(&self) -> usize {
        self.vector_dim
    }
}

use byteorder::{LittleEndian, ReadBytesExt};

pub fn read_ivecs(file_path: &str) -> std::io::Result<Vec<Vec<u32>>> {
    let file = File::open(file_path)?;
    let mut reader = BufReader::new(file);
    let mut vectors = Vec::new();

    while let Ok(dim) = reader.read_i32::<LittleEndian>() {
        let mut vec = Vec::with_capacity(dim as usize);
        for _ in 0..dim {
            let val = reader.read_i32::<LittleEndian>()?;
            vec.push(val);
        }
        vectors.push(vec);
    }

    Ok(vectors
        .into_iter()
        .map(|gt| gt.into_iter().map(|g| g as u32).collect())
        .collect())
}

fn _read_ibin(file_path: &str) -> Result<Vec<Vec<f32>>> {
    let data = std::fs::read(file_path)?;

    let num_vectors = u32::from_le_bytes(data[0..4].try_into()?) as usize;
    let vector_dim = u32::from_le_bytes(data[4..8].try_into()?) as usize;
    let start_offset = 8;
    let vector_byte_size = vector_dim * 4;

    let vectors: Vec<Vec<f32>> = (0..num_vectors)
        .into_iter()
        .map(|i| {
            let offset = start_offset + vector_byte_size * i;

            let vector: &[f32] =
                bytemuck::try_cast_slice(&data[offset..offset + vector_byte_size]).unwrap();
            vector.to_vec()
        })
        .collect();

    Ok(vectors)
}
