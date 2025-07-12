macro_rules! impl_scope {
    ($arch:ty, ($($scope:ty),+)) => {
        $(
            impl Scope for $scope {
                type Arch = $arch;
            }
        )+
    }
}

pub(crate) use impl_scope;

macro_rules! impl_projection_set_inner {
    ($arch:ty, ($($dim:ident),+), $type:ty) => {
        unsafe impl<$($dim: Dim),+> ProjectionSet<($($dim),+)> for $type
        where
            ($($dim),+): Shape,
        {
            type Arch = $arch;
        }
    }
}

pub(crate) use impl_projection_set_inner;

macro_rules! unsafe_impl_projection_set {
    ($arch:ty, $dim:tt, {$($type:ty),+}) => {
        $(
            impl_projection_set_inner!($arch, $dim, $type);
        )+
    }
}

pub(crate) use unsafe_impl_projection_set;

macro_rules! impl_projection {
    (<$space:ty, $in:ty> for $type:ty => {$arch:ty, $head:ty, $tail:ty}, {$dim:stmt}, {$idx:stmt}) => {
        impl Projection<$space, $in> for $type {
            type Arch = $arch;
            type Head = $head;
            type Tail = $tail;
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
    (<$space:ty, $in:ty> for $type:ty => {$arch:ty, $head:ty, $tail:ty}) => {
        impl Projection<$space, $in> for $type {
            type Arch = $arch;
            type Head = $head;
            type Tail = $tail;
            #[inline]
            fn dim() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                <Self::Head as Projection<$space, $in>>::dim()
                    * <Self::Tail as Projection<$space, $in>>::dim()
            }
            #[inline]
            fn idx() -> <<Self::Arch as Arch>::IndexSize as SizeType>::Unsigned {
                <Self::Head as Projection<$space, $in>>::idx()
                    + <Self::Head as Projection<$space, $in>>::dim()
                        * <Self::Tail as Projection<$space, $in>>::idx()
            }
        }
    };
}
pub(crate) use impl_projection;

macro_rules! impl_projections {
    (<$space:ty, $in:ty> for $($type:ty),+ => {$arch:ty, ($($head:ty),+), ($($tail:ty),+)}) => {
        $(impl_projection!(<$space, $in> for $type => {$arch, $head, $tail});)+
    };
}
pub(crate) use impl_projections;
