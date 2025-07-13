#![no_std]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

use crate::size_type::*;

pub struct SimtArray<T, Sc, I, L: Layout, Sh: Shape> {
    ptr: *mut T,
    layout: PhantomData<L>,
    mapping: L::Mapping<Sh>,
    scope: PhantomData<Sc>,
    state: PhantomData<I>,
}

pub struct SimtArrayViewMut<'a, T, Sc: SyncableScope, Sh: Shape, L: Layout> {
    inner: ViewMut<'a, T, Sh, L>,
    scope: PhantomData<Sc>,
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> Deref for SimtArrayViewMut<'a, T, Sc, Sh, L> {
    type Target = ViewMut<'a, T, Sh, L>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> DerefMut
    for SimtArrayViewMut<'a, T, Sc, Sh, L>
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> AsRef<ViewMut<'a, T, Sh, L>>
    for SimtArrayViewMut<'a, T, Sc, Sh, L>
{
    fn as_ref(&self) -> &ViewMut<'a, T, Sh, L> {
        &self.inner
    }
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> AsMut<ViewMut<'a, T, Sh, L>>
    for SimtArrayViewMut<'a, T, Sc, Sh, L>
{
    fn as_mut(&mut self) -> &mut ViewMut<'a, T, Sh, L> {
        &mut self.inner
    }
}

impl<'a, T, Sc: SyncableScope, L: Layout, Sh: Shape> Drop for SimtArrayViewMut<'a, T, Sc, Sh, L> {
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
    pub fn expr_mut<'a>(&'a mut self) -> ViewMut<'a, T, Sh, L> {
        unsafe { ViewMut::new_unchecked(self.ptr, self.mapping.clone()) }
    }
}

impl<T, Sc, I: Splitable + Viewable, L: Layout, D0: Dim> SimtArray<T, Sc, I, L, (D0,)>
where
    Sc: Scope,
{
    pub fn view<'a, R: RangeBounds<usize>, E, Ps>(
        &'a self,
    ) -> Option<
        View<
            'a,
            T,
            <(StepRange<R, isize>,) as ViewIndex>::Shape<(D0,)>,
            <(StepRange<R, isize>,) as ViewIndex>::Layout<L>,
        >,
    >
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
    pub fn view_mut<'a, R: RangeBounds<usize>, E, Ps>(
        &'a mut self,
    ) -> Option<
        SimtArrayViewMut<
            'a,
            T,
            Sc,
            <(StepRange<R, isize>,) as ViewIndex>::Shape<(D0,)>,
            <(StepRange<R, isize>,) as ViewIndex>::Layout<L>,
        >,
    >
    where
        Sc: SyncableScope,
        E: UnitScope<Arch = Sc::Arch>,
        Ps: ProjectionSetDim0<(D0,), E, Sc, Arch = Sc::Arch>,
    {
        let dim0 = Ps::dim0();
        let idx0 = Ps::idx0();
        if idx0.as_() > self.mapping.dim(0) {
            return None;
        }
        unsafe {
            Some(SimtArrayViewMut {
                inner: ViewMut::<_, (D0,), L>::new_unchecked(self.ptr, self.mapping.clone())
                    .into_view(StepRange {
                        range: idx0.as_()..,
                        step: dim0.as_() as isize,
                    }),
                scope: PhantomData,
            })
        }
    }
}

impl<T, Sc, I, L: Layout, D0: Dim> SimtArray<T, Sc, I, L, (D0,)>
where
    Sc: Scope + Clone,
    I: Viewable,
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

use core::{
    cell::UnsafeCell,
    marker::PhantomData,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, RangeBounds},
};

pub use archs::*;
pub use init_state::*;
use mdarray::{
    Const, Dim, Layout, Mapping, Shape, Slice, StepRange, Strided, View, ViewMut,
    expr::{Expression, enumerate, for_each},
    index::ViewIndex,
};

pub(crate) mod util;
