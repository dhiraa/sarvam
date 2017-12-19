---
layout: page
title: Misc
permalink: /deep_learning/misc/

---

#  'SAME' and 'VALID' padding in tf.nn.max_pool 

I'll give an example to make it clearer:

- x: input image of shape [2, 3], 1 channel
- "VALID" = without padding: 
  - Eg 1: max pool with 2x2 kernel, stride 2 and VALID padding.
  - Eg 2: Input width = 13, Filter width = 6, Stride = 5
```
  inputs: 1  2  3  4  5  6  7  8  9  10 11 (12 13)
          |________________|                dropped
                         |_________________|
```
- "SAME" = with zero padding:
   - Eg 1: max pool with 2x2 kernel, stride 2 and SAME padding (this is the classic way to go)
   - Eg 2: Input width = 13, Filter width = 6, Stride = 5
```
                    pad|                                      |pad
        inputs:      0 |1  2  3  4  5  6  7  8  9  10 11 12 13|0  0
                     |________________|
                                    |_________________|
                                                  |________________|
```
The output shapes are:

- valid_pad: here, no padding so the output shape is [1, 1]
- same_pad: here, we pad the image to the shape [2, 4] (with -inf and then apply max pool), so the output shape is [1, 2]

```python
x = tf.constant([[1., 2., 3.],
           [4., 5., 6.]])

x = tf.reshape(x, [1, 2, 3, 1])  # give a shape accepted by tf.nn.max_pool

valid_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
same_pad = tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

valid_pad.get_shape() == [1, 1, 1, 1]  # valid_pad is [5.]
same_pad.get_shape() == [1, 1, 2, 1]   # same_pad is  [5., 6.]
```

For the SAME padding, the output height and width are computed as:

- out_height = ceil(float(in_height) / float(strides1))
- out_width = ceil(float(in_width) / float(strides[2]))

And

For the VALID padding, the output height and width are computed as:

- out_height = ceil(float(in_height - filter_height + 1) / float(strides1))
- out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))

Notes:

- "VALID" only ever drops the right-most columns (or bottom-most rows).
- "SAME" tries to pad evenly left and right, but if the amount of columns to be added is odd, it will add the extra column to the right, as is the case in this example (the same logic applies vertically: there may be an extra row of zeros at the bottom).