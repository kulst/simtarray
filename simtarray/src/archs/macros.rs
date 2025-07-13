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

macro_rules! unsafe_impl_projection_set {
    ($arch:ty, ($($dim:ident),+$(,)?), $scope:ty, $in:ty, $type:ty) => {
        unsafe impl<$($dim: Dim),+> ProjectionSet<($($dim),+,), $scope, $in> for $type
        where
            ($($dim),+,): Shape,
        {
            type Arch = $arch;
        }
    }
}

pub(crate) use unsafe_impl_projection_set;

macro_rules! unsafe_impl_projection_sets {
    ($arch:ty, $dim:tt, $scoping:tt, {$($type:ty),+$(,)?}) => {
        $(
            unsafe_impl_projection_sets_inner!($arch, $dim, $scoping, $type);
        )+
    }
}

pub(crate) use unsafe_impl_projection_sets;

macro_rules! unsafe_impl_projection_sets_inner {
    ($arch:ty, $dim:tt, ($(<$scope:ty, $in:ty>),+$(,)?), $type:ty) => {
        $(
            unsafe_impl_projection_set!($arch, $dim, $scope, $in, $type);
        )+
    }
}

pub(crate) use unsafe_impl_projection_sets_inner;

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
