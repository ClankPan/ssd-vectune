use bit_vec::BitVec;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use itertools::Itertools;
use parking_lot::Mutex;
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use rustc_hash::FxHashMap;
use std::collections::BinaryHeap;
use std::sync::atomic;
use std::sync::atomic::AtomicUsize;
// use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(feature = "progress-bar")]
use indicatif::ProgressBar;
#[cfg(feature = "progress-bar")]
use std::sync::atomic::{self, AtomicUsize};

struct PackedNodes {
    packed_nodes_table: Vec<Mutex<bool>>,
    // packed_nodes_table: Vec<AtomicBool>,
    table_of_node_id_to_shuffled_id: Vec<u32>,
    table_of_shuffled_id_to_node_id: Vec<u32>,
    start_ids: Vec<u32>,

    #[cfg(feature = "progress-bar")]
    num_sector: usize,
}

impl PackedNodes {
    fn new(target_node_bit_vec: BitVec, rng: &mut SmallRng, window_size: usize) -> Self {
        let mut table_of_node_id_to_shuffled_id: Vec<u32> =
            (0..target_node_bit_vec.len() as u32).collect();
        let target_node_len = target_node_bit_vec.iter().filter(|&bit| bit).count();
        let mut start_ids: Vec<u32> = Vec::with_capacity(target_node_len);

        let num_sector = (target_node_len + window_size - 1) / window_size;

        table_of_node_id_to_shuffled_id.shuffle(rng);
        let table_of_shuffled_id_to_node_id: Vec<u32> = table_of_node_id_to_shuffled_id
            .iter()
            .enumerate()
            .map(|(node_id, shuffled_id)| (shuffled_id, node_id))
            .sorted()
            .map(|(_, node_id)| node_id as u32)
            .collect();
        let mut count = 0;
        let packed_nodes_table: Vec<Mutex<bool>> = table_of_shuffled_id_to_node_id
            .iter()
            .map(|node_id| {
                let is_target = target_node_bit_vec.get(*node_id as usize).unwrap();
                let is_packed = if count < num_sector && is_target {
                    count += 1;
                    start_ids.push(*node_id);
                    true
                } else {
                    !is_target // Keep non-targeted items "packed".
                };
                Mutex::new(is_packed)
            })
            .collect();

        // if start_ids.iter().all(|id| {
        //     let shuffle_id = table_of_node_id_to_shuffled_id[*id as usize] as usize;
        //     let is_packed = &packed_nodes_table[shuffle_id];
        //     *is_packed.lock()
        // }) {
        //     println!("ok start_ids")
        // } else {
        //     println!("no start_ids")
        // }

        Self {
            packed_nodes_table,
            table_of_node_id_to_shuffled_id,
            table_of_shuffled_id_to_node_id,
            start_ids,

            #[cfg(feature = "progress-bar")]
            num_sector,
        }
    }

    fn select_random_unpacked_node(&self) -> Result<u32, ()> {
        match select_random_s(&self.packed_nodes_table) {
            Ok(shuffled_id) => Ok(self.table_of_shuffled_id_to_node_id[shuffled_id as usize]),
            Err(_) => return Err(()),
        }
    }

    fn pack_node(&self, original_index: &u32) -> bool {
        let shuffled_id = &self.table_of_node_id_to_shuffled_id[*original_index as usize];
        pack_node(shuffled_id, &self.packed_nodes_table)
    }
}

fn pack_node(shuffled_index: &u32, packed_nodes_table: &Vec<Mutex<bool>>) -> bool {
    let packed_flag = &packed_nodes_table[*shuffled_index as usize];

    match packed_flag.try_lock() {
        Some(mut is_packed) => {
            if *is_packed {
                false
            } else {
                *is_packed = true;
                true
            }
        }
        None => false,
    }
}

// fn pack_node(shuffled_index: &u32, packed_nodes_table: &Vec<AtomicBool>) -> bool {
//     let packed_flag = &packed_nodes_table[*shuffled_index as usize];

//     packed_flag
//         .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
//         .is_ok()
// }

fn select_random_s(packed_nodes_table: &Vec<Mutex<bool>>) -> Result<u32, ()> {
    let mut scan_index = 0;
    loop {
        let packed_flag = &packed_nodes_table[scan_index];
        // ロックされているなら、先に取られている。
        // ロックが取れても、trueならpackされている。
        match packed_flag.try_lock() {
            Some(mut is_packed) => {
                if *is_packed {
                    scan_index += 1;
                } else {
                    let original_index = scan_index as u32;
                    *is_packed = true;
                    return Ok(original_index);
                }
            }
            None => {
                scan_index += 1;
            }
        }
        if scan_index == packed_nodes_table.len() {
            return Err(());
        }
    }
}

// fn select_random_s(packed_nodes_table: &Vec<AtomicBool>) -> Result<u32, ()> {
//     let mut scan_index = 0;
//     loop {
//         let packed_flag = &packed_nodes_table[scan_index];
//         match packed_flag.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst) {
//             Ok(_) => {
//                 let original_index = scan_index as u32;
//                 return Ok(original_index);
//             }
//             Err(_) => {
//                 scan_index += 1;
//             }
//         }
//         if scan_index == packed_nodes_table.len() {
//             return Err(());
//         }
//     }
// }

fn sector_packing<F1>(
    window_size: usize,
    get_edges: &F1,
    packed_nodes: &PackedNodes,
    start_node: u32,
) -> Vec<u32>
where
    F1: Fn(&u32) -> Vec<u32>,
{
    let mut sub_array = Vec::with_capacity(window_size);
    let mut sub_array_index = 0;
    let mut heap = KeyMaxHeap::new();

    // Pick a random, unpacked seed node s.
    sub_array.push(start_node);

    // if packed_nodes.pack_node(&start_node) {
    //     println!("start node pack is true")
    // } else {
    //     // println!("start node pack is false")
    // }

    while sub_array.len() < window_size {
        // 𝑣𝑒 ← 𝑃 [𝑖];𝑖 ← 𝑖 + 1
        let ve = &sub_array[sub_array_index];
        sub_array_index += 1;

        for u in get_edges(ve) {
            heap.increment_key(u);
            for t in get_edges(&u) {
                heap.increment_key(t);
            }
        }

        let v_max = loop {
            match heap.get_max() {
                None => {
                    // if H.empty() then Pick a random unpacked seed node 𝑣max and break.
                    // println!("not found node");
                    match packed_nodes.select_random_unpacked_node() {
                        Ok(s) => {
                            break s;
                        }
                        Err(_) => {
                            // If no unpacked nodes are found, Sector Packing is returned.
                            // println!("no unpacked nodes are found");
                            return sub_array;
                        }
                    }
                }
                Some((v_max_candidate_index, _)) => {
                    // if not D [𝑣max ] then break
                    if packed_nodes.pack_node(&v_max_candidate_index) {
                        break v_max_candidate_index;
                    }
                }
            }
        };
        sub_array.push(v_max);
    }

    sub_array
}

///
/// Reordering the arrangement to efficiently reference nodes from storage such as SSDs.
/// This algorithm is proposed in Section 4 of this [paper](https://arxiv.org/pdf/2211.12850v2.pdf).
///
pub fn merge_gorder<F1>(
    get_edges: F1,
    target_node_bit_vec: BitVec,
    window_size: usize,
    rng: &mut SmallRng,
) -> Vec<Vec<u32>>
where
    F1: Fn(&u32) -> Vec<u32> + std::marker::Sync,
{
    /* Parallel Gordering */
    // Select unpacked node randomly.
    // Scan from end to end to find nodes with the packed flag false and pick the first unpacked node found.
    // The nodes are shuffled to ensure that start nodes are randomly selected.
    // let target_node_len = target_node_bit_vec.iter().filter(|&bit| bit).count();
    let packed_nodes = PackedNodes::new(target_node_bit_vec, rng, window_size);
    let start_ids: Vec<u32> = packed_nodes.start_ids.clone();

    let pb = ProgressBar::new(1000);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green}  {msg}\n[{elapsed_precise}] {percent:>3}% {wide_bar:.cyan/blue}",
            )
            .unwrap(),
    );
    let progress_done = AtomicUsize::new(0);
    pb.set_length(start_ids.len() as u64);
    pb.set_message("merge gordering");

    let (last_id, start_ids) = start_ids.split_last().unwrap();

    let mut reordered: Vec<Vec<u32>> = start_ids
        .into_par_iter()
        .map(|start_node| {
            let res = sector_packing(window_size, &get_edges, &packed_nodes, *start_node);

            let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
            if value % 1000 == 0 {
                pb.set_position(value as u64);
            }

            res
        })
        .collect();

    let last_res = sector_packing(window_size, &get_edges, &packed_nodes, *last_id);

    let value = progress_done.fetch_add(1, atomic::Ordering::Relaxed);
    if value % 1000 == 0 {
        pb.set_position(value as u64);
    }

    // let unpacked_count = packed_nodes.packed_nodes_table.iter().filter(|is_packed| !*is_packed.lock()).count();
    // println!("unpacked_count: {}", unpacked_count);

    reordered.push(last_res);

    pb.finish();

    // let are_all_packed = reordered.iter().flatten().all(|id| !packed_nodes.pack_node(id));
    // println!("are_all_packed: {}", are_all_packed);

    reordered
}

struct KeyMaxHeap {
    heap: BinaryHeap<(u32, u32)>,
    counts: FxHashMap<u32, u32>,
}

impl KeyMaxHeap {
    fn new() -> Self {
        let mut counts = FxHashMap::default();
        counts.reserve(100);
        Self {
            heap: BinaryHeap::new(),
            counts,
        }
    }

    fn increment_key(&mut self, v: u32) {
        let count = *self.counts.get(&v).unwrap_or(&0) + 1;
        self.heap.push((count, v));
        self.counts.insert(v, count);
    }

    fn get_max(&mut self) -> Option<(u32, u32)> {
        while let Some((count, v)) = self.heap.pop() {
            if let Some(&stored_count) = self.counts.get(&v) {
                if count == stored_count {
                    return Some((v, count));
                }
            }
        }
        None
    }
}
