#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

pub struct SimtArray<T, S, I, M> {
    ptr: *mut T,
    mapping: M,
    space: S,
    init_state: I,
}

mod archs;
mod init_state;

mod size_type;

pub use init_state::*;

pub(crate) mod util;
