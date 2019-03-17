# TODO:
- Decoder input/hidden
    1. No input (Robin thinks maybe not bad) 
    2. Encoder hidden state as input, repeatedly
    3. Encoder output as input, init decoder hidden with encoder hidden, loop over input
    4. Decncoder ouptut as input (seq to seq strategy)
    5. Repeat init decoder hidden
- Regularize M_t or/both b_t: attention maps should be [0,1]
- Test to see if a type of query is causing problems
- Bad initializations
- Examine gradients/weights (M_t) to check if all modules are getting used
- Add thresholding to output
