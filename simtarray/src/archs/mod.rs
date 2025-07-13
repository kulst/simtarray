use mdarray::{Dim, Shape};
use num_traits::{One, Zero};

use crate::size_type::SizeType;

#[cfg(target_arch = "amdgpu")]
mod amdgpu;
mod macros;
#[cfg(target_arch = "nvptx64")]
pub mod nvptx;

/// The architecture trait. Examples: Nvptx, Amdgpu
pub trait Arch {
    /// Each architecture has a hierachy with different scopes. For Nvptx these
    /// scopes are for example 'Thread', 'Warp', 'Block', 'Cluster', 'Grid'
    /// The architecture of the scope type must be Self to be a valid scope
    /// for this architecture
    type Scope<S: Scope<Arch = Self>>: Scope<Arch = Self>;
    /// Each architecture has a specific BitSize it uses for indexing.
    /// For Nvptx this is _32Bit as 32 Bit integer arithmetic is much faster than
    /// 64 Bit integer arithmetic
    type IndexSize: SizeType;
}

pub trait Scope {
    type Arch: Arch;
}

/// A scope where all threads can be synchronized to each other.
pub trait SyncableScope: Scope {
    /// The function that must be used to synchronize all threads inside of this
    /// scope.
    unsafe fn sync();
}

pub trait UnitScope: Scope {}

pub trait Projection<S: Scope<Arch = Self::Arch>, O: Scope<Arch = Self::Arch>> {
    type Arch: Arch;

    type Head: Projection<S, O, Arch = Self::Arch>;
    type Tail: Projection<S, O, Arch = Self::Arch>;

    fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
    fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
}

/// # Safety
/// This trait must only be implemented for tuples of Projection's
/// so that each dimension of the arch is contained exactly once.
/// Example: For Nvptx this is implemented for example for
/// (Xyz,), (Xy, Z), (X, Yz), (Xyz, (), ()), (X, Y, Z) and many more but
/// not for (Xy, X, Z) because X is contained twice or (Xy,) because Z is missing.
pub unsafe trait ProjectionSet<Sh: Shape, S: Scope<Arch = Self::Arch>, O: Scope<Arch = Self::Arch>>
{
    type Arch: Arch;
}

pub trait ProjectionSetDim0<Sh: Shape, S: Scope<Arch = Self::Arch>, O: Scope<Arch = Self::Arch>>:
    ProjectionSet<Sh, S, O>
{
    fn dim0() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
    fn idx0() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
}

impl<P0: Projection<S, O, Arch = Self::Arch>, D0: Dim, S, O> ProjectionSetDim0<(D0,), S, O> for (P0,)
where
    S: Scope<Arch = Self::Arch>,
    O: Scope<Arch = Self::Arch>,
    (P0,): ProjectionSet<(D0,), S, O>,
{
    fn dim0() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <P0 as Projection<S, O>>::dim()
    }
    fn idx0() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <P0 as Projection<S, O>>::idx()
    }
}

// FIXME: add macro for ProjectionSetDimX implementation and implement it for
// tuples (P0,) ..= (P0, P1, ..., P5) and maybe more

impl<A: Arch, S: Scope<Arch = A>, O: Scope<Arch = A>> Projection<S, O> for () {
    type Arch = A;

    type Head = ();

    type Tail = ();

    fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <<<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned as One>::one()
    }

    fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <<<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned as Zero>::zero()
    }
}
