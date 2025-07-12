use mdarray::Dim;

use crate::size_type::_32Bit;

use super::*;
use core::arch::nvptx::*;

struct Nvptx;

impl Arch for Nvptx {
    type Space<S: Space<Arch = Self>> = S;
    type Dim<S: Space<Arch = Self>, D: Component<S, Arch = Self>> = D;

    type IndexSize = _32Bit;
}

struct GlobalSpace;

impl Space for GlobalSpace {
    type Arch = Nvptx;
}

struct SharedSpace;

impl Space for SharedSpace {
    type Arch = Nvptx;
}

impl SyncableSpace for SharedSpace {
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

macro_rules! impl_component {
    ($arch:ty, $space:ty, {$dim:stmt}, {$idx:stmt}, $type:ty) => {
        impl Component<$space> for $type {
            type Arch = $arch;
            #[inline]
            fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                // SAFETY: Can only be called on nvptx architecture and these intrinsics
                // are not unsafe per se
                $dim
            }
            #[inline]
            fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                // SAFETY: Can only be called on nvptx architecture and these intrinsics
                // are not unsafe per se
                $idx
            }
        }
    };
}

impl_component!(
    Nvptx,
    SharedSpace,
    { unsafe { _block_dim_x() as u32 } },
    { unsafe { _thread_idx_x() as u32 } },
    X
);
impl_component!(
    Nvptx,
    GlobalSpace,
    { unsafe { (_block_dim_x() * _grid_dim_x()) as u32 } },
    { unsafe { (_block_dim_x() * _block_idx_x() + _thread_idx_x()) as u32 } },
    X
);
impl_component!(
    Nvptx,
    SharedSpace,
    { unsafe { _block_dim_y() as u32 } },
    { unsafe { _thread_idx_y() as u32 } },
    Y
);
impl_component!(
    Nvptx,
    GlobalSpace,
    { unsafe { (_block_dim_y() * _grid_dim_y()) as u32 } },
    { unsafe { (_block_dim_y() * _block_idx_y() + _thread_idx_y()) as u32 } },
    Y
);
impl_component!(
    Nvptx,
    SharedSpace,
    { unsafe { _block_dim_z() as u32 } },
    { unsafe { _thread_idx_z() as u32 } },
    Z
);
impl_component!(
    Nvptx,
    GlobalSpace,
    { unsafe { (_block_dim_z() * _grid_dim_z()) as u32 } },
    { unsafe { (_block_dim_z() * _block_idx_z() + _thread_idx_z()) as u32 } },
    Z
);

macro_rules! impl_component_set {
    ($arch:ty, ($($space:ty),+), $head:ty, $tail:ty, $augmented_sets:tt, $elements:tt, EmptySet) => {
        $(
            impl_component_set_inner!($arch, $space, $head, $tail, $augmented_sets, $elements, EmptySet);
        )+
    };
    ($arch:ty, ($($space:ty),+), $head:ty, $tail:ty, $augmented_sets:tt, $elements:tt, $type:ty) => {
        $(
            impl_component_set_inner!($arch, $space, $head, $tail, $augmented_sets, $elements, $type);
        )+
    }
}

macro_rules! impl_component_set_inner {
    ($arch:ty, $space:ty, $head:ty, $tail:ty, ($($augmented:ty),+), ($($element:ty),+), EmptySet) => {
        $(
            impl ComponentSet<$space, $element> for EmptySet {
                type Arch = $arch;

                type Head = $head;
                type Tail = $tail;
                type Augmented = $augmented;
                #[inline]
                fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                    1u32
                }
                #[inline]
                fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                    0u32
                }
            }
        )+
    };
    ($arch:ty, $space:ty, $head:ty, $tail:ty, ($($augmented:ty),+), ($($element:ty),+), $type:ty) => {
        $(
            impl ComponentSet<$space, $element> for $type {
                type Arch = $arch;

                type Head = $head;
                type Tail = $tail;
                type Augmented = $augmented;
            }
        )+
    };
}

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    NoComponent,
    EmptySet,
    (EmptySet, X, Y, Z),
    (NoComponent, X, Y, Z),
    EmptySet
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    EmptySet,
    (X, Xy, Xz),
    (NoComponent, Y, Z),
    X
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Y,
    EmptySet,
    (Y, Xy, Yz),
    (NoComponent, X, Z),
    Y
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Z,
    EmptySet,
    (Z, Xz, Yz),
    (NoComponent, X, Y),
    Z
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Y,
    (Xy, Xyz),
    (NoComponent, Z),
    Xy
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Z,
    (Xz, Xyz),
    (NoComponent, Y),
    Xz
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Y,
    Z,
    (Yz, Xyz),
    (NoComponent, X),
    Yz
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Yz,
    (Xyz),
    (NoComponent),
    Xyz
);

impl<A: Dim> PaddedComponentPartition<(A,)> for (Xyz,)
where
    (A,): Shape,
{
    type Arch = Nvptx;
}

macro_rules! impl_component_bundle {
    ($arch:ty, $dim:tt, {$($type:ty),+}) => {
        $(
            impl_component_bundle_inner!($arch, $dim, $type);
        )+
    }
}

macro_rules! impl_component_bundle_inner {
    ($arch:ty, ($($dim:ident),+), $type:ty) => {
        impl<$($dim: Dim),+> PaddedComponentPartition<($($dim),+)> for $type
        where
            ($($dim),+): Shape,
        {
            type Arch = $arch;
        }
    }
}

impl_component_bundle!(
    Nvptx,
    (A, B),
    {
        (Xyz, NoComponent),
        (Yz, X),
        (Xz, Y),
        (Z, Xy),
        (Xy, Z),
        (Y, Xz),
        (X, Yz),
        (NoComponent, Xyz)
    }
);

impl_component_bundle!(
    Nvptx,
    (A, B, C),
    {
        (Xyz, NoComponent, NoComponent),
        (Yz, X, NoComponent),
        (Yz, NoComponent, X),
        (Xz, Y, NoComponent),
        (Z, Xy, NoComponent),
        (Z, Y, X),
        (Xz, NoComponent, Y),
        (Z, X, Y),
        (Z, NoComponent, Xy),
        (Xy, Z, NoComponent),
        (Y, Xz, NoComponent),
        (Y, Z, X),
        (X, Yz, NoComponent),
        (NoComponent, Xyz, NoComponent),
        (NoComponent, Yz, X),
        (X, Z, Y),
        (NoComponent, Xz, Y),
        (NoComponent, Z, Xy),
        (Xy, NoComponent, Z),
        (Y, X, Z),
        (Y, NoComponent, Xz),
        (X, Y, Z),
        (NoComponent, Xy, Z),
        (NoComponent, Y, Xz),
        (X, NoComponent, Yz),
        (NoComponent, X, Yz),
        (NoComponent, NoComponent, Xyz)
    }
);

impl_component_bundle!(
    Nvptx,
    (A, B, C, D),
    {
        (Xyz, NoComponent, NoComponent, NoComponent),
        (Yz, X, NoComponent, NoComponent),
        (Yz, NoComponent, X, NoComponent),
        (Yz, NoComponent, NoComponent, X),
        (Xz, Y, NoComponent, NoComponent),
        (Z, Xy, NoComponent, NoComponent),
        (Z, Y, X, NoComponent),
        (Z, Y, NoComponent, X),
        (Xz, NoComponent, Y, NoComponent),
        (Z, X, Y, NoComponent),
        (Z, NoComponent, Xy, NoComponent),
        (Z, NoComponent, Y, X),
        (Xz, NoComponent, NoComponent, Y),
        (Z, X, NoComponent, Y),
        (Z, NoComponent, X, Y),
        (Z, NoComponent, NoComponent, Xy),
        (Xy, Z, NoComponent, NoComponent),
        (Y, Xz, NoComponent, NoComponent),
        (Y, Z, X, NoComponent),
        (Y, Z, NoComponent, X),
        (X, Yz, NoComponent, NoComponent),
        (NoComponent, Xyz, NoComponent, NoComponent),
        (NoComponent, Yz, X, NoComponent),
        (NoComponent, Yz, NoComponent, X),
        (X, Z, Y, NoComponent),
        (NoComponent, Xz, Y, NoComponent),
        (NoComponent, Z, Xy, NoComponent),
        (NoComponent, Z, Y, X),
        (X, Z, NoComponent, Y),
        (NoComponent, Xz, NoComponent, Y),
        (NoComponent, Z, X, Y),
        (NoComponent, Z, NoComponent, Xy),
        (Xy, NoComponent, Z, NoComponent),
        (Y, X, Z, NoComponent),
        (Y, NoComponent, Xz, NoComponent),
        (Y, NoComponent, Z, X),
        (X, Y, Z, NoComponent),
        (NoComponent, Xy, Z, NoComponent),
        (NoComponent, Y, Xz, NoComponent),
        (NoComponent, Y, Z, X),
        (X, NoComponent, Yz, NoComponent),
        (NoComponent, X, Yz, NoComponent),
        (NoComponent, NoComponent, Xyz, NoComponent),
        (NoComponent, NoComponent, Yz, X),
        (X, NoComponent, Z, Y),
        (NoComponent, X, Z, Y),
        (NoComponent, NoComponent, Xz, Y),
        (NoComponent, NoComponent, Z, Xy),
        (Xy, NoComponent, NoComponent, Z),
        (Y, X, NoComponent, Z),
        (Y, NoComponent, X, Z),
        (Y, NoComponent, NoComponent, Xz),
        (X, Y, NoComponent, Z),
        (NoComponent, Xy, NoComponent, Z),
        (NoComponent, Y, X, Z),
        (NoComponent, Y, NoComponent, Xz),
        (X, NoComponent, Y, Z),
        (NoComponent, X, Y, Z),
        (NoComponent, NoComponent, Xy, Z),
        (NoComponent, NoComponent, Y, Xz),
        (X, NoComponent, NoComponent, Yz),
        (NoComponent, X, NoComponent, Yz),
        (NoComponent, NoComponent, X, Yz),
        (NoComponent, NoComponent, NoComponent, Xyz)
    }
);

impl_component_bundle!(
    Nvptx,
    (A, B, C, D, E),
    {
        (Xyz, NoComponent, NoComponent, NoComponent, NoComponent),
        (Yz, X, NoComponent, NoComponent, NoComponent),
        (Yz, NoComponent, X, NoComponent, NoComponent),
        (Yz, NoComponent, NoComponent, X, NoComponent),
        (Yz, NoComponent, NoComponent, NoComponent, X),
        (Xz, Y, NoComponent, NoComponent, NoComponent),
        (Z, Xy, NoComponent, NoComponent, NoComponent),
        (Z, Y, X, NoComponent, NoComponent),
        (Z, Y, NoComponent, X, NoComponent),
        (Z, Y, NoComponent, NoComponent, X),
        (Xz, NoComponent, Y, NoComponent, NoComponent),
        (Z, X, Y, NoComponent, NoComponent),
        (Z, NoComponent, Xy, NoComponent, NoComponent),
        (Z, NoComponent, Y, X, NoComponent),
        (Z, NoComponent, Y, NoComponent, X),
        (Xz, NoComponent, NoComponent, Y, NoComponent),
        (Z, X, NoComponent, Y, NoComponent),
        (Z, NoComponent, X, Y, NoComponent),
        (Z, NoComponent, NoComponent, Xy, NoComponent),
        (Z, NoComponent, NoComponent, Y, X),
        (Xz, NoComponent, NoComponent, NoComponent, Y),
        (Z, X, NoComponent, NoComponent, Y),
        (Z, NoComponent, X, NoComponent, Y),
        (Z, NoComponent, NoComponent, X, Y),
        (Z, NoComponent, NoComponent, NoComponent, Xy),
        (Xy, Z, NoComponent, NoComponent, NoComponent),
        (Y, Xz, NoComponent, NoComponent, NoComponent),
        (Y, Z, X, NoComponent, NoComponent),
        (Y, Z, NoComponent, X, NoComponent),
        (Y, Z, NoComponent, NoComponent, X),
        (X, Yz, NoComponent, NoComponent, NoComponent),
        (NoComponent, Xyz, NoComponent, NoComponent, NoComponent),
        (NoComponent, Yz, X, NoComponent, NoComponent),
        (NoComponent, Yz, NoComponent, X, NoComponent),
        (NoComponent, Yz, NoComponent, NoComponent, X),
        (X, Z, Y, NoComponent, NoComponent),
        (NoComponent, Xz, Y, NoComponent, NoComponent),
        (NoComponent, Z, Xy, NoComponent, NoComponent),
        (NoComponent, Z, Y, X, NoComponent),
        (NoComponent, Z, Y, NoComponent, X),
        (X, Z, NoComponent, Y, NoComponent),
        (NoComponent, Xz, NoComponent, Y, NoComponent),
        (NoComponent, Z, X, Y, NoComponent),
        (NoComponent, Z, NoComponent, Xy, NoComponent),
        (NoComponent, Z, NoComponent, Y, X),
        (X, Z, NoComponent, NoComponent, Y),
        (NoComponent, Xz, NoComponent, NoComponent, Y),
        (NoComponent, Z, X, NoComponent, Y),
        (NoComponent, Z, NoComponent, X, Y),
        (NoComponent, Z, NoComponent, NoComponent, Xy),
        (Xy, NoComponent, Z, NoComponent, NoComponent),
        (Y, X, Z, NoComponent, NoComponent),
        (Y, NoComponent, Xz, NoComponent, NoComponent),
        (Y, NoComponent, Z, X, NoComponent),
        (Y, NoComponent, Z, NoComponent, X),
        (X, Y, Z, NoComponent, NoComponent),
        (NoComponent, Xy, Z, NoComponent, NoComponent),
        (NoComponent, Y, Xz, NoComponent, NoComponent),
        (NoComponent, Y, Z, X, NoComponent),
        (NoComponent, Y, Z, NoComponent, X),
        (X, NoComponent, Yz, NoComponent, NoComponent),
        (NoComponent, X, Yz, NoComponent, NoComponent),
        (NoComponent, NoComponent, Xyz, NoComponent, NoComponent),
        (NoComponent, NoComponent, Yz, X, NoComponent),
        (NoComponent, NoComponent, Yz, NoComponent, X),
        (X, NoComponent, Z, Y, NoComponent),
        (NoComponent, X, Z, Y, NoComponent),
        (NoComponent, NoComponent, Xz, Y, NoComponent),
        (NoComponent, NoComponent, Z, Xy, NoComponent),
        (NoComponent, NoComponent, Z, Y, X),
        (X, NoComponent, Z, NoComponent, Y),
        (NoComponent, X, Z, NoComponent, Y),
        (NoComponent, NoComponent, Xz, NoComponent, Y),
        (NoComponent, NoComponent, Z, X, Y),
        (NoComponent, NoComponent, Z, NoComponent, Xy),
        (Xy, NoComponent, NoComponent, Z, NoComponent),
        (Y, X, NoComponent, Z, NoComponent),
        (Y, NoComponent, X, Z, NoComponent),
        (Y, NoComponent, NoComponent, Xz, NoComponent),
        (Y, NoComponent, NoComponent, Z, X),
        (X, Y, NoComponent, Z, NoComponent),
        (NoComponent, Xy, NoComponent, Z, NoComponent),
        (NoComponent, Y, X, Z, NoComponent),
        (NoComponent, Y, NoComponent, Xz, NoComponent),
        (NoComponent, Y, NoComponent, Z, X),
        (X, NoComponent, Y, Z, NoComponent),
        (NoComponent, X, Y, Z, NoComponent),
        (NoComponent, NoComponent, Xy, Z, NoComponent),
        (NoComponent, NoComponent, Y, Xz, NoComponent),
        (NoComponent, NoComponent, Y, Z, X),
        (X, NoComponent, NoComponent, Yz, NoComponent),
        (NoComponent, X, NoComponent, Yz, NoComponent),
        (NoComponent, NoComponent, X, Yz, NoComponent),
        (NoComponent, NoComponent, NoComponent, Xyz, NoComponent),
        (NoComponent, NoComponent, NoComponent, Yz, X),
        (X, NoComponent, NoComponent, Z, Y),
        (NoComponent, X, NoComponent, Z, Y),
        (NoComponent, NoComponent, X, Z, Y),
        (NoComponent, NoComponent, NoComponent, Xz, Y),
        (NoComponent, NoComponent, NoComponent, Z, Xy),
        (Xy, NoComponent, NoComponent, NoComponent, Z),
        (Y, X, NoComponent, NoComponent, Z),
        (Y, NoComponent, X, NoComponent, Z),
        (Y, NoComponent, NoComponent, X, Z),
        (Y, NoComponent, NoComponent, NoComponent, Xz),
        (X, Y, NoComponent, NoComponent, Z),
        (NoComponent, Xy, NoComponent, NoComponent, Z),
        (NoComponent, Y, X, NoComponent, Z),
        (NoComponent, Y, NoComponent, X, Z),
        (NoComponent, Y, NoComponent, NoComponent, Xz),
        (X, NoComponent, Y, NoComponent, Z),
        (NoComponent, X, Y, NoComponent, Z),
        (NoComponent, NoComponent, Xy, NoComponent, Z),
        (NoComponent, NoComponent, Y, X, Z),
        (NoComponent, NoComponent, Y, NoComponent, Xz),
        (X, NoComponent, NoComponent, Y, Z),
        (NoComponent, X, NoComponent, Y, Z),
        (NoComponent, NoComponent, X, Y, Z),
        (NoComponent, NoComponent, NoComponent, Xy, Z),
        (NoComponent, NoComponent, NoComponent, Y, Xz),
        (X, NoComponent, NoComponent, NoComponent, Yz),
        (NoComponent, X, NoComponent, NoComponent, Yz),
        (NoComponent, NoComponent, X, NoComponent, Yz),
        (NoComponent, NoComponent, NoComponent, X, Yz),
        (NoComponent, NoComponent, NoComponent, NoComponent, Xyz)
    }
);

impl_component_bundle!(
    Nvptx,
    (A, B, C, D, E, F),
    {
        (Xyz, NoComponent, NoComponent, NoComponent, NoComponent, NoComponent),
        (Yz, X, NoComponent, NoComponent, NoComponent, NoComponent),
        (Yz, NoComponent, X, NoComponent, NoComponent, NoComponent),
        (Yz, NoComponent, NoComponent, X, NoComponent, NoComponent),
        (Yz, NoComponent, NoComponent, NoComponent, X, NoComponent),
        (Yz, NoComponent, NoComponent, NoComponent, NoComponent, X),
        (Xz, Y, NoComponent, NoComponent, NoComponent, NoComponent),
        (Z, Xy, NoComponent, NoComponent, NoComponent, NoComponent),
        (Z, Y, X, NoComponent, NoComponent, NoComponent),
        (Z, Y, NoComponent, X, NoComponent, NoComponent),
        (Z, Y, NoComponent, NoComponent, X, NoComponent),
        (Z, Y, NoComponent, NoComponent, NoComponent, X),
        (Xz, NoComponent, Y, NoComponent, NoComponent, NoComponent),
        (Z, X, Y, NoComponent, NoComponent, NoComponent),
        (Z, NoComponent, Xy, NoComponent, NoComponent, NoComponent),
        (Z, NoComponent, Y, X, NoComponent, NoComponent),
        (Z, NoComponent, Y, NoComponent, X, NoComponent),
        (Z, NoComponent, Y, NoComponent, NoComponent, X),
        (Xz, NoComponent, NoComponent, Y, NoComponent, NoComponent),
        (Z, X, NoComponent, Y, NoComponent, NoComponent),
        (Z, NoComponent, X, Y, NoComponent, NoComponent),
        (Z, NoComponent, NoComponent, Xy, NoComponent, NoComponent),
        (Z, NoComponent, NoComponent, Y, X, NoComponent),
        (Z, NoComponent, NoComponent, Y, NoComponent, X),
        (Xz, NoComponent, NoComponent, NoComponent, Y, NoComponent),
        (Z, X, NoComponent, NoComponent, Y, NoComponent),
        (Z, NoComponent, X, NoComponent, Y, NoComponent),
        (Z, NoComponent, NoComponent, X, Y, NoComponent),
        (Z, NoComponent, NoComponent, NoComponent, Xy, NoComponent),
        (Z, NoComponent, NoComponent, NoComponent, Y, X),
        (Xz, NoComponent, NoComponent, NoComponent, NoComponent, Y),
        (Z, X, NoComponent, NoComponent, NoComponent, Y),
        (Z, NoComponent, X, NoComponent, NoComponent, Y),
        (Z, NoComponent, NoComponent, X, NoComponent, Y),
        (Z, NoComponent, NoComponent, NoComponent, X, Y),
        (Z, NoComponent, NoComponent, NoComponent, NoComponent, Xy),
        (Xy, Z, NoComponent, NoComponent, NoComponent, NoComponent),
        (Y, Xz, NoComponent, NoComponent, NoComponent, NoComponent),
        (Y, Z, X, NoComponent, NoComponent, NoComponent),
        (Y, Z, NoComponent, X, NoComponent, NoComponent),
        (Y, Z, NoComponent, NoComponent, X, NoComponent),
        (Y, Z, NoComponent, NoComponent, NoComponent, X),
        (X, Yz, NoComponent, NoComponent, NoComponent, NoComponent),
        (NoComponent, Xyz, NoComponent, NoComponent, NoComponent, NoComponent),
        (NoComponent, Yz, X, NoComponent, NoComponent, NoComponent),
        (NoComponent, Yz, NoComponent, X, NoComponent, NoComponent),
        (NoComponent, Yz, NoComponent, NoComponent, X, NoComponent),
        (NoComponent, Yz, NoComponent, NoComponent, NoComponent, X),
        (X, Z, Y, NoComponent, NoComponent, NoComponent),
        (NoComponent, Xz, Y, NoComponent, NoComponent, NoComponent),
        (NoComponent, Z, Xy, NoComponent, NoComponent, NoComponent),
        (NoComponent, Z, Y, X, NoComponent, NoComponent),
        (NoComponent, Z, Y, NoComponent, X, NoComponent),
        (NoComponent, Z, Y, NoComponent, NoComponent, X),
        (X, Z, NoComponent, Y, NoComponent, NoComponent),
        (NoComponent, Xz, NoComponent, Y, NoComponent, NoComponent),
        (NoComponent, Z, X, Y, NoComponent, NoComponent),
        (NoComponent, Z, NoComponent, Xy, NoComponent, NoComponent),
        (NoComponent, Z, NoComponent, Y, X, NoComponent),
        (NoComponent, Z, NoComponent, Y, NoComponent, X),
        (X, Z, NoComponent, NoComponent, Y, NoComponent),
        (NoComponent, Xz, NoComponent, NoComponent, Y, NoComponent),
        (NoComponent, Z, X, NoComponent, Y, NoComponent),
        (NoComponent, Z, NoComponent, X, Y, NoComponent),
        (NoComponent, Z, NoComponent, NoComponent, Xy, NoComponent),
        (NoComponent, Z, NoComponent, NoComponent, Y, X),
        (X, Z, NoComponent, NoComponent, NoComponent, Y),
        (NoComponent, Xz, NoComponent, NoComponent, NoComponent, Y),
        (NoComponent, Z, X, NoComponent, NoComponent, Y),
        (NoComponent, Z, NoComponent, X, NoComponent, Y),
        (NoComponent, Z, NoComponent, NoComponent, X, Y),
        (NoComponent, Z, NoComponent, NoComponent, NoComponent, Xy),
        (Xy, NoComponent, Z, NoComponent, NoComponent, NoComponent),
        (Y, X, Z, NoComponent, NoComponent, NoComponent),
        (Y, NoComponent, Xz, NoComponent, NoComponent, NoComponent),
        (Y, NoComponent, Z, X, NoComponent, NoComponent),
        (Y, NoComponent, Z, NoComponent, X, NoComponent),
        (Y, NoComponent, Z, NoComponent, NoComponent, X),
        (X, Y, Z, NoComponent, NoComponent, NoComponent),
        (NoComponent, Xy, Z, NoComponent, NoComponent, NoComponent),
        (NoComponent, Y, Xz, NoComponent, NoComponent, NoComponent),
        (NoComponent, Y, Z, X, NoComponent, NoComponent),
        (NoComponent, Y, Z, NoComponent, X, NoComponent),
        (NoComponent, Y, Z, NoComponent, NoComponent, X),
        (X, NoComponent, Yz, NoComponent, NoComponent, NoComponent),
        (NoComponent, X, Yz, NoComponent, NoComponent, NoComponent),
        (NoComponent, NoComponent, Xyz, NoComponent, NoComponent, NoComponent),
        (NoComponent, NoComponent, Yz, X, NoComponent, NoComponent),
        (NoComponent, NoComponent, Yz, NoComponent, X, NoComponent),
        (NoComponent, NoComponent, Yz, NoComponent, NoComponent, X),
        (X, NoComponent, Z, Y, NoComponent, NoComponent),
        (NoComponent, X, Z, Y, NoComponent, NoComponent),
        (NoComponent, NoComponent, Xz, Y, NoComponent, NoComponent),
        (NoComponent, NoComponent, Z, Xy, NoComponent, NoComponent),
        (NoComponent, NoComponent, Z, Y, X, NoComponent),
        (NoComponent, NoComponent, Z, Y, NoComponent, X),
        (X, NoComponent, Z, NoComponent, Y, NoComponent),
        (NoComponent, X, Z, NoComponent, Y, NoComponent),
        (NoComponent, NoComponent, Xz, NoComponent, Y, NoComponent),
        (NoComponent, NoComponent, Z, X, Y, NoComponent),
        (NoComponent, NoComponent, Z, NoComponent, Xy, NoComponent),
        (NoComponent, NoComponent, Z, NoComponent, Y, X),
        (X, NoComponent, Z, NoComponent, NoComponent, Y),
        (NoComponent, X, Z, NoComponent, NoComponent, Y),
        (NoComponent, NoComponent, Xz, NoComponent, NoComponent, Y),
        (NoComponent, NoComponent, Z, X, NoComponent, Y),
        (NoComponent, NoComponent, Z, NoComponent, X, Y),
        (NoComponent, NoComponent, Z, NoComponent, NoComponent, Xy),
        (Xy, NoComponent, NoComponent, Z, NoComponent, NoComponent),
        (Y, X, NoComponent, Z, NoComponent, NoComponent),
        (Y, NoComponent, X, Z, NoComponent, NoComponent),
        (Y, NoComponent, NoComponent, Xz, NoComponent, NoComponent),
        (Y, NoComponent, NoComponent, Z, X, NoComponent),
        (Y, NoComponent, NoComponent, Z, NoComponent, X),
        (X, Y, NoComponent, Z, NoComponent, NoComponent),
        (NoComponent, Xy, NoComponent, Z, NoComponent, NoComponent),
        (NoComponent, Y, X, Z, NoComponent, NoComponent),
        (NoComponent, Y, NoComponent, Xz, NoComponent, NoComponent),
        (NoComponent, Y, NoComponent, Z, X, NoComponent),
        (NoComponent, Y, NoComponent, Z, NoComponent, X),
        (X, NoComponent, Y, Z, NoComponent, NoComponent),
        (NoComponent, X, Y, Z, NoComponent, NoComponent),
        (NoComponent, NoComponent, Xy, Z, NoComponent, NoComponent),
        (NoComponent, NoComponent, Y, Xz, NoComponent, NoComponent),
        (NoComponent, NoComponent, Y, Z, X, NoComponent),
        (NoComponent, NoComponent, Y, Z, NoComponent, X),
        (X, NoComponent, NoComponent, Yz, NoComponent, NoComponent),
        (NoComponent, X, NoComponent, Yz, NoComponent, NoComponent),
        (NoComponent, NoComponent, X, Yz, NoComponent, NoComponent),
        (NoComponent, NoComponent, NoComponent, Xyz, NoComponent, NoComponent),
        (NoComponent, NoComponent, NoComponent, Yz, X, NoComponent),
        (NoComponent, NoComponent, NoComponent, Yz, NoComponent, X),
        (X, NoComponent, NoComponent, Z, Y, NoComponent),
        (NoComponent, X, NoComponent, Z, Y, NoComponent),
        (NoComponent, NoComponent, X, Z, Y, NoComponent),
        (NoComponent, NoComponent, NoComponent, Xz, Y, NoComponent),
        (NoComponent, NoComponent, NoComponent, Z, Xy, NoComponent),
        (NoComponent, NoComponent, NoComponent, Z, Y, X),
        (X, NoComponent, NoComponent, Z, NoComponent, Y),
        (NoComponent, X, NoComponent, Z, NoComponent, Y),
        (NoComponent, NoComponent, X, Z, NoComponent, Y),
        (NoComponent, NoComponent, NoComponent, Xz, NoComponent, Y),
        (NoComponent, NoComponent, NoComponent, Z, X, Y),
        (NoComponent, NoComponent, NoComponent, Z, NoComponent, Xy),
        (Xy, NoComponent, NoComponent, NoComponent, Z, NoComponent),
        (Y, X, NoComponent, NoComponent, Z, NoComponent),
        (Y, NoComponent, X, NoComponent, Z, NoComponent),
        (Y, NoComponent, NoComponent, X, Z, NoComponent),
        (Y, NoComponent, NoComponent, NoComponent, Xz, NoComponent),
        (Y, NoComponent, NoComponent, NoComponent, Z, X),
        (X, Y, NoComponent, NoComponent, Z, NoComponent),
        (NoComponent, Xy, NoComponent, NoComponent, Z, NoComponent),
        (NoComponent, Y, X, NoComponent, Z, NoComponent),
        (NoComponent, Y, NoComponent, X, Z, NoComponent),
        (NoComponent, Y, NoComponent, NoComponent, Xz, NoComponent),
        (NoComponent, Y, NoComponent, NoComponent, Z, X),
        (X, NoComponent, Y, NoComponent, Z, NoComponent),
        (NoComponent, X, Y, NoComponent, Z, NoComponent),
        (NoComponent, NoComponent, Xy, NoComponent, Z, NoComponent),
        (NoComponent, NoComponent, Y, X, Z, NoComponent),
        (NoComponent, NoComponent, Y, NoComponent, Xz, NoComponent),
        (NoComponent, NoComponent, Y, NoComponent, Z, X),
        (X, NoComponent, NoComponent, Y, Z, NoComponent),
        (NoComponent, X, NoComponent, Y, Z, NoComponent),
        (NoComponent, NoComponent, X, Y, Z, NoComponent),
        (NoComponent, NoComponent, NoComponent, Xy, Z, NoComponent),
        (NoComponent, NoComponent, NoComponent, Y, Xz, NoComponent),
        (NoComponent, NoComponent, NoComponent, Y, Z, X),
        (X, NoComponent, NoComponent, NoComponent, Yz, NoComponent),
        (NoComponent, X, NoComponent, NoComponent, Yz, NoComponent),
        (NoComponent, NoComponent, X, NoComponent, Yz, NoComponent),
        (NoComponent, NoComponent, NoComponent, X, Yz, NoComponent),
        (NoComponent, NoComponent, NoComponent, NoComponent, Xyz, NoComponent),
        (NoComponent, NoComponent, NoComponent, NoComponent, Yz, X),
        (X, NoComponent, NoComponent, NoComponent, Z, Y),
        (NoComponent, X, NoComponent, NoComponent, Z, Y),
        (NoComponent, NoComponent, X, NoComponent, Z, Y),
        (NoComponent, NoComponent, NoComponent, X, Z, Y),
        (NoComponent, NoComponent, NoComponent, NoComponent, Xz, Y),
        (NoComponent, NoComponent, NoComponent, NoComponent, Z, Xy),
        (Xy, NoComponent, NoComponent, NoComponent, NoComponent, Z),
        (Y, X, NoComponent, NoComponent, NoComponent, Z),
        (Y, NoComponent, X, NoComponent, NoComponent, Z),
        (Y, NoComponent, NoComponent, X, NoComponent, Z),
        (Y, NoComponent, NoComponent, NoComponent, X, Z),
        (Y, NoComponent, NoComponent, NoComponent, NoComponent, Xz),
        (X, Y, NoComponent, NoComponent, NoComponent, Z),
        (NoComponent, Xy, NoComponent, NoComponent, NoComponent, Z),
        (NoComponent, Y, X, NoComponent, NoComponent, Z),
        (NoComponent, Y, NoComponent, X, NoComponent, Z),
        (NoComponent, Y, NoComponent, NoComponent, X, Z),
        (NoComponent, Y, NoComponent, NoComponent, NoComponent, Xz),
        (X, NoComponent, Y, NoComponent, NoComponent, Z),
        (NoComponent, X, Y, NoComponent, NoComponent, Z),
        (NoComponent, NoComponent, Xy, NoComponent, NoComponent, Z),
        (NoComponent, NoComponent, Y, X, NoComponent, Z),
        (NoComponent, NoComponent, Y, NoComponent, X, Z),
        (NoComponent, NoComponent, Y, NoComponent, NoComponent, Xz),
        (X, NoComponent, NoComponent, Y, NoComponent, Z),
        (NoComponent, X, NoComponent, Y, NoComponent, Z),
        (NoComponent, NoComponent, X, Y, NoComponent, Z),
        (NoComponent, NoComponent, NoComponent, Xy, NoComponent, Z),
        (NoComponent, NoComponent, NoComponent, Y, X, Z),
        (NoComponent, NoComponent, NoComponent, Y, NoComponent, Xz),
        (X, NoComponent, NoComponent, NoComponent, Y, Z),
        (NoComponent, X, NoComponent, NoComponent, Y, Z),
        (NoComponent, NoComponent, X, NoComponent, Y, Z),
        (NoComponent, NoComponent, NoComponent, X, Y, Z),
        (NoComponent, NoComponent, NoComponent, NoComponent, Xy, Z),
        (NoComponent, NoComponent, NoComponent, NoComponent, Y, Xz),
        (X, NoComponent, NoComponent, NoComponent, NoComponent, Yz),
        (NoComponent, X, NoComponent, NoComponent, NoComponent, Yz),
        (NoComponent, NoComponent, X, NoComponent, NoComponent, Yz),
        (NoComponent, NoComponent, NoComponent, X, NoComponent, Yz),
        (NoComponent, NoComponent, NoComponent, NoComponent, X, Yz),
        (NoComponent, NoComponent, NoComponent, NoComponent, NoComponent, Xyz)
    }
);
