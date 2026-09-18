# `nullcal.time_frequency_transform`

## Compatibility note

The JAX WDM implementation removes the legacy buffer helpers `assign_wdata`,
`pack_wave`, `pack_wave_inverse`, `unpack_wave_inverse`, `dx_assign_loop`,
`dx_unpack_loop`, `dx_unpack_loop_quadrature`, `unpack_time_wave_helper`,
`unpack_time_wave_helper_compact`, `pack_wave_time_helper`, and
`pack_wave_time_helper_compact`. These implementation details were never
re-exported from `nullcal.time_frequency_transform`; callers that imported them
directly from their defining submodules should migrate to the public transform
functions documented below.

::: nullcal.time_frequency_transform
    options:
        docstring_style: google
        show_source: true
        show_root_heading: true
        show_object_full_path: true
        members_order: source
        filters:
            - '!^_'
