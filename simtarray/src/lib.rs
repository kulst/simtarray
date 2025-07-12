#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

pub struct SimtArray<T, S, I, M> {
    ptr: *mut T,
    mapping: M,
    space: S,
    init_state: I,
}

impl<D0, T, M, S, I> SimtArray<T, S, I, M>
where
    D0: Dim,
    M: Mapping<Shape = (D0,)>,
    S: Space,
    I: InitState,
{
    fn link<CS0: ComponentSet<S, C0, Arch = S::Arch>, C0: Component<S, Arch = S::Arch>>()
    where
        (CS0,): PaddedComponentPartition<M::Shape, Arch = S::Arch>,
    {
    }
}

mod archs;
mod init_state;

mod size_type;

use archs::{Component, ComponentSet, PaddedComponentPartition, Space};
pub use init_state::*;
use mdarray::{Dim, Mapping};

pub(crate) mod util;
