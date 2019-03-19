# TODO:
- Init input to decoder could be improved
- Improve x_t (query attention mapping)
- Test to see if a type of query is causing problems
- Examine gradients/weights (M_t) to check if all modules are getting used
- Add thresholding to output
- Better debugging visualizations for flow of info
- Plot the loss/learning curves
- Follow the composition graph from the output

# Done:
- Loss function  (MSE,NLL,...) 
- Init a_t with noise for better learning
- Input to decode: Started as 0s, temporarly init with random and out->in
- Added attention to the decoder (started with hidden_n, then hidden_end_query, final soft hidden_n)
- Regularize M_t or/both b_t: attention maps should be [0,1] (Tried relu, tanh, using softmax now)

# Random
- And, Or, Id removal does not seem to impact training
