#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

pub struct SimtArray<T, Sc, I, L: Layout, Sh: Shape> {
    ptr: *mut T,
    layout: PhantomData<L>,
    mapping: L::Mapping<Sh>,
    scope: PhantomData<Sc>,
    state: PhantomData<I>,
}

pub struct SimtArrayMutRef<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> {
    view: Option<(*mut T, L::Mapping<Sh>)>,
    layout: PhantomData<&'a mut L>,
    scope: PhantomData<Sc>,
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> SimtArrayMutRef<'a, T, Sc, L, Sh> {
    pub fn expr_mut<'b>(&'b mut self) -> Option<ViewMut<'b, T, Sh, L>>
    where
        'a: 'b,
    {
        self.view
            .as_ref()
            .map(|view| unsafe { ViewMut::new_unchecked(view.0, view.1.clone()) })
    }
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> Drop for SimtArrayMutRef<'a, T, Sc, L, Sh> {
    fn drop(&mut self) {
        unsafe { Sc::sync() };
    }
}

impl<T, Sc, I, L: Layout, Sh: Shape> SimtArray<T, Sc, I, L, Sh>
where
    Sc: Scope,
    I: State,
{
    /// # Safety
    pub unsafe fn new_unchecked(ptr: *mut T, mapping: L::Mapping<Sh>) -> Self {
        Self {
            ptr,
            mapping,
            layout: PhantomData,
            scope: PhantomData,
            state: PhantomData,
        }
    }
}

impl<T, Sc, I: Splitable, L: Layout, Sh: Shape> SimtArray<T, Sc, I, L, Sh>
where
    Sc: Scope,
    I: Viewable,
{
    pub fn expr<'a>(&'a self) -> View<'a, T, Sh, L> {
        unsafe { View::new_unchecked(self.ptr as *const T, self.mapping.clone()) }
    }
    // pub fn expr_mut<'a>(&'a mut self) -> ViewMut<'a, T, Sh, L> {
    //     unsafe { ViewMut::new_unchecked(self.ptr, self.mapping.clone()) }
    // }
}
type Unsigned<Sc> = <<<Sc as Scope>::Arch as Arch>::IndexSize as SizeType>::Unsigned;

impl<T, Sc, I: Splitable + Viewable, L: Layout, D0: Dim> SimtArray<T, Sc, I, L, (D0,)>
where
    Sc: Scope,
{
    pub fn view_with_limited_quantity<'a, E, Ps>(
        &'a self,
        quantity: Unsigned<Sc>,
    ) -> Option<View<'a, T, (usize,), Strided>>
    where
        E: Scope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) {
            return None;
        }
        if idx0 >= quantity {
            return None;
        }
        let quantity = dim0.min(quantity);

        unsafe {
            Some(
                View::<_, (D0,), L>::new_unchecked(self.ptr, self.mapping.clone()).into_view(
                    StepRange {
                        range: idx0.as_()..,
                        step: quantity.as_() as isize,
                    },
                ),
            )
        }
    }
    pub fn view_mut_with_limited_quantity<'a, E, Ps>(
        &'a mut self,
        quantity: Unsigned<Sc>,
    ) -> SimtArrayMutRef<'a, T, Sc, Strided, (usize,)>
    where
        Sc: SyncableScope,
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) || idx0 >= quantity {
            return SimtArrayMutRef {
                view: None,
                scope: PhantomData,
                layout: PhantomData,
            };
        }
        let quantity = dim0.min(quantity);
        let view_mut = unsafe {
            ViewMut::<_, (D0,), L>::new_unchecked(self.ptr, self.mapping.clone()).into_view(
                StepRange {
                    range: idx0.as_()..,
                    step: quantity.as_() as isize,
                },
            )
        };
        SimtArrayMutRef {
            view: Some(view_mut.into_raw_parts()),
            layout: PhantomData,
            scope: PhantomData,
        }
    }
    pub fn view<'a, E, Ps>(&'a self) -> Option<View<'a, T, (usize,), Strided>>
    where
        E: Scope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) {
            return None;
        }

        unsafe {
            Some(
                View::<_, (D0,), L>::new_unchecked(self.ptr, self.mapping.clone()).into_view(
                    StepRange {
                        range: idx0.as_()..,
                        step: dim0.as_() as isize,
                    },
                ),
            )
        }
    }
    pub fn view_mut<'a, 'b, 'c, E, Ps>(
        &'a mut self,
    ) -> SimtArrayMutRef<'a, T, Sc, Strided, (usize,)>
    where
        'a: 'b,
        'b: 'c,
        Sc: SyncableScope,
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) {
            return SimtArrayMutRef {
                view: None,
                scope: PhantomData,
                layout: PhantomData,
            };
        }
        let view_mut = unsafe {
            ViewMut::<_, (D0,), L>::new_unchecked(self.ptr, self.mapping.clone()).into_view(
                StepRange {
                    range: idx0.as_()..,
                    step: dim0.as_() as isize,
                },
            )
        };
        SimtArrayMutRef {
            view: Some(view_mut.into_raw_parts()),
            layout: PhantomData,
            scope: PhantomData,
        }
    }
}

impl<T, Sc, I, L: Layout, D0: Dim> SimtArray<T, Sc, I, L, (D0,)>
where
    Sc: Scope,
    I: Splitable,
{
    /// # Safety
    /// Must be called in a kernel uniform control flow state
    pub unsafe fn write_once<E, Ps, F>(
        self,
        f: F,
    ) -> Option<SimtArray<T, Sc, FinallySplit, Strided, (usize,)>>
    where
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
        F: FnMut((D0,)) -> T,
    {
        unsafe { self.write_once_inner::<_, Ps, _>(f) }
    }
    /// # Safety
    /// Must be called in a kernel uniform control flow state
    unsafe fn write_once_inner<E, Ps, F>(
        self,
        mut f: F,
    ) -> Option<SimtArray<T, Sc, FinallySplit, Strided, (usize,)>>
    where
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
        F: FnMut((D0,)) -> T,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) {
            return None;
        }
        let view_parts = unsafe {
            View::<_, (D0,), L>::new_unchecked(
                self.ptr as *const UnsafeCell<MaybeUninit<T>>,
                self.mapping,
            )
        }
        .into_view(StepRange {
            range: idx0.as_()..,
            step: dim0.as_() as isize,
        })
        .into_raw_parts();
        let view = unsafe {
            ViewMut::<_, _, Strided>::new_unchecked(
                view_parts.0 as *mut MaybeUninit<T>,
                view_parts.1,
            )
        };
        for_each(enumerate(view), |(idx, item)| {
            item.write(f((D0::from_size(idx),)));
        });
        Some(SimtArray {
            ptr: view_parts.0 as *mut T,
            scope: PhantomData,
            state: PhantomData,
            layout: PhantomData,
            mapping: view_parts.1,
        })
    }

    /// # Safety
    /// Must be called in a kernel uniform control flow state
    pub unsafe fn init_with<E, Ps, F>(self, f: F) -> SimtArray<T, Sc, Init, L, (D0,)>
    where
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
        F: FnMut((D0,)) -> T,
        Sc: SyncableScope,
    {
        let out = SimtArray {
            ptr: self.ptr,
            layout: PhantomData,
            mapping: self.mapping.clone(),
            scope: PhantomData,
            state: PhantomData,
        };
        unsafe {
            self.write_once_inner::<_, Ps, _>(f);
            <Sc as SyncableScope>::sync();
        }
        out
    }
}

mod archs;
mod init_state;

mod size_type;

use core::{cell::UnsafeCell, marker::PhantomData, mem::MaybeUninit};

pub use archs::*;
pub use init_state::*;
use mdarray::{
    Dim, Layout, Mapping, Shape, StepRange, Strided, View, ViewMut,
    expr::{enumerate, for_each},
};
pub use size_type::*;

pub(crate) mod util;
