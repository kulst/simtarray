use crate::util::Sealed;

pub trait State: Sealed {}
pub trait Splitable: State {}
pub trait Viewable: State {}

pub struct Init;

pub struct Uninit;

pub struct FinallySplit;

impl State for Init {}
impl State for Uninit {}
impl State for FinallySplit {}

impl Splitable for Init {}
impl Splitable for Uninit {}

impl Viewable for Init {}
impl Viewable for FinallySplit {}
