use core::marker::PhantomData;

use mdarray::Shape;
use num_traits::{One, Zero};

use crate::size_type::SizeType;

#[cfg(target_arch = "nvptx64")]
mod nvptx;

#[cfg(target_arch = "amdgpu")]
mod amdgpu;

pub trait Arch {
    type Space<S: Space<Arch = Self>>: Space<Arch = Self>;
    type Dim<S: Space<Arch = Self>, D: Component<S, Arch = Self>>: Component<S, Arch = Self>;

    type IndexSize: SizeType;
}

pub trait Space {
    type Arch: Arch;
}

pub trait SyncableSpace: Space {
    unsafe fn sync();
}

pub trait Component<S: Space<Arch = Self::Arch>> {
    type Arch: Arch;

    fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
    fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned;
}

pub trait ComponentSet<S: Space<Arch = Self::Arch>, E: Component<S, Arch = Self::Arch>> {
    type Arch: Arch;

    type Head: Component<S, Arch = Self::Arch>;
    type Tail: ComponentSet<S, E, Arch = Self::Arch>;
    /// Augmented ComponentSet after adding E to Self
    type Augmented: ComponentSet<S, NoComponent, Arch = Self::Arch>;

    fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        Self::Head::dim() * Self::Tail::dim()
    }
    fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        Self::Head::idx() + Self::Head::dim() * Self::Tail::idx()
    }
}

pub struct NoComponent;

impl<A: Arch, S: Space<Arch = A>> Component<S> for NoComponent {
    type Arch = A;

    fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <<<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned as One>::one()
    }

    fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
        <<<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned as Zero>::zero()
    }
}

pub struct EmptySet;

impl<A: Arch, S: Space<Arch = A>, E: Component<S, Arch = A>> ComponentSet<S, E> for EmptySet
where
    (E,): ComponentSet<S, NoComponent, Arch = A>,
{
    type Arch = A;

    type Head = NoComponent;

    type Tail = EmptySet;

    type Augmented = (E,);
}

pub trait PaddedComponentPartition<S: Shape> {
    type Arch: Arch;
}
