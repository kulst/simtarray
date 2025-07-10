use num_traits::PrimInt;

use crate::util::Sealed;

pub struct _8Bit;
pub struct _16Bit;
pub struct _32Bit;
pub struct _64Bit;
pub struct Size;

#[allow(private_bounds)]
pub trait SizeType: Sealed {
    type Unsigned: AsUsize;
    type Signed: AsIsize;
}

#[allow(private_bounds)]
pub trait AsUsize: 'static + PrimInt + Sealed + Sync + Send {
    fn as_(self) -> usize;
}

#[allow(private_bounds)]
pub trait AsIsize: 'static + PrimInt + Sealed + Sync + Send {
    fn as_(self) -> isize;
}

impl AsUsize for usize {
    #[inline]
    fn as_(self) -> usize {
        self
    }
}

impl AsIsize for isize {
    #[inline]
    fn as_(self) -> isize {
        self
    }
}

macro_rules! impl_as_size {
    (($($U:ty),+), ($($I:ty),+)) => {
        $(impl AsUsize for $U {
            #[inline] fn as_(self) -> usize { self as usize }
        })+
        $(impl AsIsize for $I {
            #[inline] fn as_(self) -> isize { self as isize }
        })+
    };
}

#[cfg(target_pointer_width = "16")]
impl_as_size!((u8, u16), (i8, i16));
#[cfg(target_pointer_width = "32")]
impl_as_size!((u8, u16, u32), (i8, i16, i32));
#[cfg(target_pointer_width = "64")]
impl_as_size!((u8, u16, u32, u64), (i8, i16, i32, i64));

impl SizeType for _8Bit {
    type Unsigned = u8;
    type Signed = i8;
}

impl SizeType for _16Bit {
    type Unsigned = u16;
    type Signed = i16;
}

impl SizeType for _32Bit {
    type Unsigned = u32;
    type Signed = i32;
}

impl SizeType for _64Bit {
    type Unsigned = u64;
    type Signed = i64;
}

impl SizeType for Size {
    type Unsigned = usize;
    type Signed = isize;
}

macro_rules! impl_sealed {
    ($($T:ty),+) => {
        $(impl Sealed for $T {})+
    };
}

impl_sealed!(
    u8, u16, u32, u64, usize, i8, i16, i32, i64, isize, _8Bit, _16Bit, _32Bit, _64Bit, Size
);
