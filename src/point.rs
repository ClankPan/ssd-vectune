#[cfg(feature = "simd")]
use std::simd::f32x4;

use serde::{Deserialize, Serialize};
use vectune::PointInterface;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Point(Vec<f32>);
impl Point {
    // fn to_f32_vec(&self) -> Vec<f32> {
    //     self.0.to_vec()
    // }
    // fn from_f32_vec(a: Vec<f32>) -> Self {
    //     Point(a.into_iter().collect())
    // }

    pub const DIM: usize = 96;
}

impl PointInterface for Point {
    #[cfg(not(feature = "simd"))]
    fn distance(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| {
                let c = a - b;
                c * c
            })
            .sum::<f32>()
            .sqrt()
    }

    #[cfg(feature = "simd")]
    fn distance(&self, other: &Self) -> f32 {
        assert_eq!(self.0.len(), other.0.len());

        let mut sum = f32x4::splat(0.0);
        let chunks = self.0.chunks_exact(4).zip(other.0.chunks_exact(4));

        for (a_chunk, b_chunk) in chunks {
            let a_simd = f32x4::from_slice(a_chunk);
            let b_simd = f32x4::from_slice(b_chunk);
            let diff = a_simd - b_simd;
            sum += diff * diff;
        }

        // Convert SIMD vector sum to an array and sum its elements
        let simd_sum: f32 = sum.to_array().iter().sum();

        // Handle remaining elements
        let remainder_start = self.0.len() - self.0.len() % 4;
        let remainder_sum: f32 = self.0[remainder_start..]
            .iter()
            .zip(&other.0[remainder_start..])
            .map(|(a, b)| {
                let diff = a - b;
                diff * diff
            })
            .sum();

        // Calculate the total sum and then the square root
        (simd_sum + remainder_sum).sqrt()
    }

    fn dim() -> u32 {
        96
    }

    fn add(&self, other: &Self) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .zip(other.to_f32_vec())
                .map(|(x, y)| x + y)
                .collect(),
        )
    }
    fn div(&self, divisor: &usize) -> Self {
        Point::from_f32_vec(
            self.to_f32_vec()
                .into_iter()
                .map(|v| v / *divisor as f32)
                .collect(),
        )
    }

    fn zero() -> Self {
        Point::from_f32_vec(vec![0.0; Point::dim() as usize])
    }

    fn to_f32_vec(&self) -> Vec<f32> {
        self.0.iter().copied().collect()
    }
    fn from_f32_vec(a: Vec<f32>) -> Self {
        Point(a.into_iter().collect())
    }
}

// fn dot_product(vec1: &Point, vec2: &Point) -> f32 {
//     let mut result = 0.0;
//     for i in 0..Point::DIM {
//         result += vec1.0[i] * vec2.0[i];
//     }
//     result
// }

// fn norm(vec: &Point) -> f32 {
//     let mut result = 0.0;
//     for i in 0..Point::DIM {
//         result += vec.0[i] * vec.0[i];
//     }
//     result.sqrt()
// }

// fn cosine_similarity(vec1: &Point, vec2: &Point) -> f32 {
//     let dot = dot_product(vec1, vec2);
//     let norm1 = norm(vec1);
//     let norm2 = norm(vec2);

//     if norm1 == 0.0 || norm2 == 0.0 {
//         return 0.0;
//     }

//     dot / (norm1 * norm2)
// }
