# C API

The library has one public header, `include/mr_aha/mr_aha.h`, with a pure C surface
(`extern "C"`). Everything below is generated from its Doxygen comments. A worked example is in
[C / C++](../getting-started/c.md) and in
[`examples/example_aha.cu`](https://github.com/mikecjz/MR_AHA/blob/main/examples/example_aha.cu).

Array shapes are column-major, as everywhere in the documentation
(see [Conventions](../guide/conventions.md)).

```{doxygenfile} mr_aha.h
:project: mr_aha
```
