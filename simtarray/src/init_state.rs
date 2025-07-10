use crate::util::Sealed;

pub trait InitState: Sealed {}

pub struct Init;

pub struct Uninit;

impl InitState for Init {}
impl InitState for Uninit {}
