# Binning

_Data binning_, also called _discrete binning_ or _bucketing_, is a data pre-processing technique used to reduce the effects of minor observation errors. The original data values which fall into a given small interval, a bin, are replaced by a value representative of that interval, often the central value. It is a form of quantization.

There are two classes are implemented here for data binning: `BinsEncoder` and `BinsDiscretizer`.

* `BinsDiscretizer` is used to split all values of real feature vector into some bins.
* `BinsEncoder` is used to encode each bin with some value.

## TODO list
- [ ] Check and add asserts
- [ ] Make `encode_bins` parameter more flexible may be?
- [ ] Process empty bins borders correctly
- [ ] Process NaNs correctly
