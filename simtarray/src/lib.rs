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
    S: Scope,
    I: InitState,
{
    /// SAFETY: Must be called in a kernel uniform control flow state
    unsafe fn write_once<O, P0, F>(self, f: F)
    where
        O: UnitScope<Arch = S::Arch>,
        P0: Projection<O, S, Arch = S::Arch>,
        (P0,): ProjectionSet<M::Shape, Arch = S::Arch>,
        F: FnMut((D0,)) -> T,
    {
        todo!()
    }

    /// SAFETY: Must be called in a kernel uniform control flow state
    unsafe fn init_with<O, P0, F>(self, f: F) -> SimtArray<T, S, Init, M>
    where
        O: UnitScope<Arch = S::Arch>,
        P0: Projection<O, S, Arch = S::Arch>,
        (P0,): ProjectionSet<M::Shape, Arch = S::Arch>,
        F: FnMut((D0,)) -> T,
        S: SyncableScope,
    {
        unsafe { <S as SyncableScope>::sync() };
        todo!()
    }
}

mod archs;
mod init_state;

mod size_type;

use archs::{Projection, ProjectionSet, Scope, SyncableScope, UnitScope};
pub use init_state::*;
use mdarray::{Dim, Mapping};

pub(crate) mod util;
