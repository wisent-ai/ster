# How a computation on tensors fails

Every declared way a computation Ster runs through Candle fails, whichever
command runs it: a forward pass, a training step's backward pass and update,
and a tensor read or written. Candle hands back one error type; which of its
variants it is, never its words, decides the cause: the accelerator refusing
the operation (out of memory, or a driver failing the kernel, exit 69), a
device with no kernel for the dtype the weights are mapped at, tensors Ster
built that do not fit each other or a view a kernel does not take (defects
of Ster), a gradient through an inference kernel, and a tensor file Candle
cannot read or write. The step a refusal names is what the site was
computing, or the steps the error's context names when it reached the end
of the command undeclared. How a refusal reads and which status it exits with
is on [How Ster fails](../errors.md).

Everything below the marker is
`ster errors list --markdown --page docs/guide/errors/tensor.md`;
`tests/errors/pages.rs` and `tests/errors/catalogue.sh` refuse a page that
differs from it.

<!-- generated: ster errors list --markdown -->
