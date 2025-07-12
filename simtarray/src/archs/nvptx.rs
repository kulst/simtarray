use mdarray::Dim;

use crate::size_type::_32Bit;

use super::macros::*;
use super::*;
use core::arch::nvptx::*;

struct Nvptx;

impl Arch for Nvptx {
    type Scope<S: Scope<Arch = Self>> = S;

    type IndexSize = _32Bit;
}

struct Grid;
struct Block;
struct Thread;

impl_scope!(Nvptx, (Grid, Block, Thread));

impl UnitScope for Thread {}

impl SyncableScope for Block {
    #[inline]
    unsafe fn sync() {
        unsafe { _syncthreads() }
    }
}

struct X;
struct Y;
struct Z;
struct Xy;
struct Xz;
struct Xyz;
struct Yz;

impl_projection!(
    <Thread, Block> for X => { Nvptx, X, () },
    { unsafe { _block_dim_x() as u32 } },
    { unsafe { _thread_idx_x() as u32 } }
);
impl_projection!(
    <Thread, Block> for Y => { Nvptx, Y, () },
    { unsafe { _block_dim_y() as u32 } },
    { unsafe { _thread_idx_y() as u32 } }
);
impl_projection!(
    <Thread, Block> for Z => { Nvptx, Z, () },
    { unsafe { _block_dim_z() as u32 } },
    { unsafe { _thread_idx_z() as u32 } }
);
impl_projection!(
    <Thread, Grid> for X => { Nvptx, X, () },
    { unsafe { (_block_dim_x() * _grid_dim_x()) as u32 } },
    { unsafe { (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as u32 } }
);
impl_projection!(
    <Thread, Grid> for Y => { Nvptx, Y, () },
    { unsafe { (_block_dim_y() * _grid_dim_y()) as u32 } },
    { unsafe { (_block_dim_y() * _block_idx_y() + _thread_idx_y()) as u32 } }
);
impl_projection!(
    <Thread, Grid> for Z => { Nvptx, Z, () },
    { unsafe { (_block_dim_z() * _grid_dim_z()) as u32 } },
    { unsafe { (_block_dim_z() * _block_idx_z() + _thread_idx_z()) as u32 } }
);
impl_projection!(
    <Block, Grid> for X => { Nvptx, X, () },
    { unsafe { _grid_dim_x() as u32 } },
    { unsafe { _block_idx_x() as u32 } }
);
impl_projection!(
    <Block, Grid> for Y => { Nvptx, Y, () },
    { unsafe { _grid_dim_y() as u32 } },
    { unsafe { _block_idx_y() as u32 } }
);
impl_projection!(
    <Block, Grid> for Z => { Nvptx, Z, () },
    { unsafe { _grid_dim_z() as u32 } },
    { unsafe { _block_idx_z() as u32 } }
);
impl_projections!(<Thread,Block> for Xy, Xz, Yz, Xyz => {Nvptx, (X, X, Y, X), (Y, Z, Z, Yz)});
impl_projections!(<Thread,Grid> for Xy, Xz, Yz, Xyz => {Nvptx, (X, X, Y, X), (Y, Z, Z, Yz)});
impl_projections!(<Block,Grid> for Xy, Xz, Yz, Xyz => {Nvptx, (X, X, Y, X), (Y, Z, Z, Yz)});

unsafe impl<D1: Dim> ProjectionSet<(D1,)> for (Xyz,)
where
    (D1,): Shape,
{
    type Arch = Nvptx;
}

unsafe_impl_projection_set!(Nvptx, (D1, D2), {
    (Xyz, ()),
    (Yz, X),
    (Xz, Y),
    (Z, Xy),
    (Xy, Z),
    (Y, Xz),
    (X, Yz),
    ((), Xyz)
});

unsafe_impl_projection_set!(Nvptx, (D1, D2, D3), {
    (Xyz, (), ()),
    (Yz, X, ()),
    (Yz, (), X),
    (Xz, Y, ()),
    (Z, Xy, ()),
    (Z, Y, X),
    (Xz, (), Y),
    (Z, X, Y),
    (Z, (), Xy),
    (Xy, Z, ()),
    (Y, Xz, ()),
    (Y, Z, X),
    (X, Yz, ()),
    ((), Xyz, ()),
    ((), Yz, X),
    (X, Z, Y),
    ((), Xz, Y),
    ((), Z, Xy),
    (Xy, (), Z),
    (Y, X, Z),
    (Y, (), Xz),
    (X, Y, Z),
    ((), Xy, Z),
    ((), Y, Xz),
    (X, (), Yz),
    ((), X, Yz),
    ((), (), Xyz)
});

// FIXME: For more dimensions we should probably use proc macros
