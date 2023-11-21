This code is used to explain one of my papers, which has been submitted to STOCHASTIC ENVIROMENTAL RESEARCH AND RISK ASSSSMENT. In codes，0 represents " inrush water ", 1 represents " Xujiazhuang limestone water ", 2 represents " Ordovician limestone water ",
 3 represents " sandstone water ",  and 4 represents " goaf water ".For the learning framework we used, 
it is not possible to directly remove the label '0' from the output. 
However, in practice, our custom loss function effectively ignores it, so the model outputs a value so small that its probability can be disregarded.
 This code provides a way to process the results into four outputs. 
To activate this feature, simply uncomment the provided code：


    #zero_index = np.where(category_encoder.categories_[0] == 0)[0][0]
    #predictions[:, zero_index] = 0
    #predictions /= np.sum(predictions, axis=1, keepdims=True)
