//! Entry points that exist so every Stream Engine client links, but have no
//! implementation behind them.
//!
//! A stub is a safe `extern "C"` function: it reads none of its arguments, so
//! nothing a caller passes can make it unsound, and it needs no `# Safety`
//! section. Its parameters are typed with "any bit pattern is valid" types
//! (`*const c_void`, integers, `f32`) whatever the C header says, because the
//! C types only matter to a callee that reads them.

/// Define `extern "C"` stubs returning `TOBII_ERROR_NOT_SUPPORTED`.
macro_rules! not_supported {
    ($( $(#[$m:meta])* fn $name:ident ( $( $arg:ident : $ty:ty ),* $(,)? ); )+) => { $(
        $(#[$m])*
        #[doc = concat!(
            "`", stringify!($name), "`: not implemented by libtobii.so; returns ",
            "`TOBII_ERROR_NOT_SUPPORTED` without reading its arguments."
        )]
        #[unsafe(no_mangle)]
        pub extern "C" fn $name($( $arg: $ty ),*) -> $crate::status::Status {
            $( let _ = $arg; )*
            tracing::trace!(concat!(stringify!($name), ": not supported"));
            $crate::status::TOBII_ERROR_NOT_SUPPORTED
        }
    )+ };
}

pub(crate) use not_supported;
