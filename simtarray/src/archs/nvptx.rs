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

            fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                // SAFETY: Can only be called on nvptx architecture and these intrinsics
                // are not unsafe per se
                $dim
            }

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
    ($arch:ty, ($($space:ty),+), $head:ty, $tail:ty, $augmented_sets:tt, $elements:tt, ()) => {
        $(
            impl_component_set_inner!($arch, $space, $head, $tail, $augmented_sets, $elements, ());
        )+
    };
    ($arch:ty, ($($space:ty),+), $head:ty, $tail:ty, $augmented_sets:tt, $elements:tt, $type:ty) => {
        $(
            impl_component_set_inner!($arch, $space, $head, $tail, $augmented_sets, $elements, $type);
        )+
    }
}

macro_rules! impl_component_set_inner {
    ($arch:ty, $space:ty, $head:ty, $tail:ty, ($($augmented:ty),+), ($($element:ty),+), ()) => {
        $(
            impl ComponentSet<$space, $element> for () {
                type Arch = $arch;

                type Head = $head;
                type Tail = $tail;
                type Augmented = $augmented;
                fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                    1u32
                }
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
    NoComponent<Nvptx>,
    (),
    ((), X, Y, Z),
    (NoComponent<Nvptx>, X, Y, Z),
    ()
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    (),
    (X, Xy, Xz),
    (NoComponent<Nvptx>, Y, Z),
    X
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Y,
    (),
    (Y, Xy, Yz),
    (NoComponent<Nvptx>, X, Z),
    Y
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Z,
    (),
    (Z, Xz, Yz),
    (NoComponent<Nvptx>, X, Y),
    Z
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Y,
    (Xy, Xyz),
    (NoComponent<Nvptx>, Z),
    Xy
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Z,
    (Xz, Xyz),
    (NoComponent<Nvptx>, Y),
    Xz
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    Y,
    Z,
    (Yz, Xyz),
    (NoComponent<Nvptx>, X),
    Yz
);

impl_component_set!(
    Nvptx,
    (SharedSpace, GlobalSpace),
    X,
    Yz,
    (Xyz),
    (NoComponent<Nvptx>),
    Xyz
);
