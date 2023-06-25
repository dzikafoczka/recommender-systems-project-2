# Neural Network Recommender - Project II

Department of Mathematics and Computer Science, Adam Mickiewicz University, 2023

Author: Paweł Łączkowski

## Project description

The main goal of the project was to create a neural network recommender for hotel data (room recommendations) whose @HR10 metric score would be higher than the amazon recommender.

To complete this task, the following subtasks had to be accomplished:
1. Preparation of data (data preprocessing for hybrid models).
2. Preparation of user features (for hybrid models).
3. Preparation of item features (for hybrid models).
4. Creating a recommender (prepare network architectures).
5. Tuning the recommender.
6. Evaluation of the recommender.

## Preparation of training data (data preprocessing)

The entire process of preparing data can be found in the file `project_1_data_preparation`.

It covers the creation of new features based on available data, such as a `length_of_stay` or `weekend_stay`, and the bucketing of important features.

## Preparation of user features

The task was to extract numerical features for users from the data that could be used to train the recommendation model.

### User features

User features are distributions of individual item features for all items with which that user interacted.

In general, the data shows what proportion of all user interactions were, for example, `Christmas term` or `Weekend Stay` interactions.

## Preparation of item features

The task was to extract numerical features for items from the data that could be used to train the recommendation model.

Item features were prepared in a similar way to user features due to the later use of the scalar product of features to train the recommender.

### Item features

Item features are One-Hot Encoded.

## Creating a recommender.

The recomenders are included in file - `project_2_recommender_and_evaluation - final`.

The recomender fitting works as follows:
1. Prepare users and items features (for hybrid models).
2. Generate negative interactions.
3. Prepare input for training machine learning model.
4. Train machine learning model.

The recomendations work as follows:
1. Prepare users and items features (for hybrid models).
2. Calculate score (predict values) for input data.
3. Get highest scores from list.
4. Return `n` items with highest score.

My experiments consisted of testing recommenders based on several different architectures.

Throughout the workflow, I prepared 12 different architectures, 7 based solely on embeddings layers and 5 on embeddings layers with the addition of content-based features that I prepared earlier for Project 1.

## Non-hybrid Models

### [1] One fully-connected layer without bias (200+ evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(fc): Linear(in_features=6, out_features=1, bias=False)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

This architecture uses multiplied element-by-element embeddings of the users and the items, and then passes them to a fully connected layer. Finally, sigmoid function is applied.

**The best result (with additional manual tuning): 0.248812**

This neural network is really simple, but it also gives pretty good results.

### [2] Two fully-connected layers without bias (50+ evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(fc1): Linear(in_features=6, out_features=8, bias=False)`
* `(fc2): Linear(in_features=8, out_features=1, bias=False)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

**The best result: 0.179009**

This neural network performs significantly worse than its simpler counterpart, so I decided to give up on this architecture after 50+ evaluations.

### [3, 4, 5, 6] Three/Four/Five/Six fully-connected layers without bias:

This architectures performs significantly worse than its simpler counterparts, so I decided to give up on all of this architectures.

### [7] GMF + MLP (150 + evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(gmf_user_embedding): Embedding(14503, 6)`
* `(gmf_item_embedding): Embedding(763, 6)`
* `(mlp_user_embedding): Embedding(14503, 6)`
* `(mlp_item_embedding): Embedding(763, 6)`
* `(mlp_fc1): Linear(in_features=12, out_features=16, bias=True)`
* `(mlp_fc2): Linear(in_features=16, out_features=8, bias=True)`
* `(fc): Linear(in_features=14, out_features=1, bias=False)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

This architectures uses two separate pairs or embeddings for users and items.
First pair (`gmf_user_embedding` and `gmf_item_embedding`) is multiplied element-by-element.
Second pair (`mlp_user_embedding` and `mlp_item_embedding`) is concatenated into one vector and then passed through two fully-connected layers.
Then, two results mentioned above are then concatenated and passed through a final fully-connected layer. Finally, sigmoid function is applied.

This architecture performs really solid, but in the end it achieved a final result slightly worse than the first one.

**The best result: 0.240665**

## Hybrid Models

### [1, 2] One/Three fully connected-layers + content-based features:

This architectures performs really badly, and the results are below any reasonable level, so I decided to give up on this architectures.

### [3] GMF + MLP + Content-based #1 (150+ evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(gmf_user_embedding): Embedding(14503, 6)`
* `(gmf_item_embedding): Embedding(763, 6)`
* `(mlp_user_embedding): Embedding(14503, 6)`
* `(mlp_item_embedding): Embedding(763, 6)`
* `(mlp_fc1): Linear(in_features=12, out_features=16, bias=True)`
* `(mlp_fc2): Linear(in_features=16, out_features=8, bias=True)`
* `(content_based): Linear(in_features=24, out_features=8, bias=True)`
* `(cb_fc1): Linear(in_features=48, out_features=16, bias=True)`
* `(cb_fc2): Linear(in_features=16, out_features=8, bias=True)`
* `(fc1): Linear(in_features=30, out_features=16, bias=True)`
* `(fc2): Linear(in_features=16, out_features=1, bias=True)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

This architecture combines GMF+MLP architecture with content-based features.

Content-based features are provided in two ways:

- as multipled vectors (element-by-element) of users and items features.
- as concatenated vectors of users and items features.

Multipled vectors of features are passed through one fully-connected layer without any activation function.

Concatenated vectors are passed through two fully-connected layers with relu activation functions.

In the end all results are concatenated and passed through two final fully-connected layers. Finally, sigmoid function is applied.

This architecture works really well and gives overall better results than the first (non-hybrid) neural network.

**The best result: 0.255601**

### [4] GMF + MLP + Content-based #2 (75+ evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(gmf_user_embedding): Embedding(14503, 6)`
* `(gmf_item_embedding): Embedding(763, 6)`
* `(mlp_user_embedding): Embedding(14503, 6)`
* `(mlp_item_embedding): Embedding(763, 6)`
* `(mlp_fc1): Linear(in_features=12, out_features=16, bias=True)`
* `(mlp_fc2): Linear(in_features=16, out_features=8, bias=True)`
* `(content_based): Linear(in_features=24, out_features=8, bias=True)`
* `(fc1): Linear(in_features=22, out_features=16, bias=True)`
* `(fc2): Linear(in_features=16, out_features=1, bias=True)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

This architecture combines GMF+MLP architecture with content-based features.

The difference between **GMF + MLP + Content-based #2** and this architecture is that this architecture uses only multipled vectors of users and items features.

This architecture works really well and in the end got the best score among all architectures.

**The best result (with additional manual tuning): 0.265105**

### [5] GMF + MLP + Content-based #3 (200+ evals):
* `(item_embedding): Embedding(763, 6)`
* `(user_embedding): Embedding(14503, 6)`
* `(gmf_user_embedding): Embedding(14503, 6)`
* `(gmf_item_embedding): Embedding(763, 6)`
* `(mlp_user_embedding): Embedding(14503, 6)`
* `(mlp_item_embedding): Embedding(763, 6)`
* `(mlp_fc1): Linear(in_features=12, out_features=16, bias=True)`
* `(mlp_fc2): Linear(in_features=16, out_features=8, bias=True)`
* `(content_based): Linear(in_features=24, out_features=8, bias=True)`
* `(content_based_sep): Linear(in_features=48, out_features=8, bias=True)`
* `(fc1): Linear(in_features=30, out_features=16, bias=True)`
* `(fc2): Linear(in_features=16, out_features=1, bias=True)`

**Attention! The number of neurons in the architecture depends on the embedding dimension (in the example it is set to 6).**

This architecture combines GMF+MLP architecture with content-based features.

It is simmilar to **GMF + MLP + Content-based #1** architecture.

The difference is that this architecture does not pass the connected vectors through fully-connected layer and does not use the activation function.

This architecture works really well and in the end got the second best score among all architectures.

**The best result (with additional manual tuning): 0.262729**

## Tuning the recommender.

Recommender tuning involves properly training the machine learning models used to predict scores.

Many tunings have been performed for all the models mentioned above (only a small part of the tuning process is included in files).

Most of the tuning results are in the catalog `results_csv`.

I also hand-tuned the best models for maximum performance and highest scores.

### Tuned parameters

#### n_neg_per_pos
`n_neg_per_pos` - number of negative interactions per positive one (1-10).

#### n_epochs
`n_epochs` - number of training epochs (1-100).

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).

#### SVRCBUIRecommender

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).
2. `C` - regularization parameter.
3. `kernel` - kernel type.

#### RandomForestCBUIRecommender

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).
2. `n_estimators` - number of trees in the forest.
3. `max_depth` - maximum depth of the tree.
4. `min_samples_split` - minimum number of samples required to split an internal node.

#### XGBoostCBUIRecommender

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).
2. `n_estimators` - number of boosting stages to perform. 
3. `max_depth` - maximum depth of the individual regression estimators.
4. `min_samples_split` - minimum number of samples required to split an internal node.
5. `learning_rate` - learning rate parameter.

#### NeuralNetworkCBUIRecommender

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).
2. `epochs` - number of epochs of the training process.

#### CatBoostRegressorCBUIRecommender

Tuned hyperparameters:
1. `n_neg_per_pos` - number of negative interactions per positive one (1-10).
2. `iterations` - maximum number of trees.
3. `l2_leaf_reg` - regularization parameter.
4. `learning_rate` - learning rate parameter.
5. `depth` - depth of the tree.

## Evaluation of the recommender.

The final results are as follows (models whose results were unsatisfactory were omitted):

#### User and Item Features #1

| **Recommender**                  | **HR@10** |
|----------------------------------|-----------|
| LinearRegressionCBUIRecommender  | 0.209776  |
| RandomForestCBUIRecommender      | 0.187373  |
| XGBoostCBUIRecommender           | 0.210115  |
| CatBoostRegressorCBUIRecommender | 0.217583  |
| AmazonRecommender                | 0.185336  |

#### User and Item Features #2

| **Recommender**                  | **HR@10** |
|----------------------------------|-----------|
| LinearRegressionCBUIRecommender  | 0.071623  |
| XGBoostCBUIRecommender           | 0.102172  |
| CatBoostRegressorCBUIRecommender | 0.202308  |
| AmazonRecommender                | 0.185336  |

## Conclusion

`User and Item Features #1` turned out to be much better than the `User and Item Features #2` and produced satisfying results on most models.

The CatBoost model achieved the best score of 0.217583, beating amazon's recommender score by 0.032247 points.

## Project requirements

Project is written in python. 

Requires:
1. Python version 3.8 or higher.
2. [Anaconda](https://www.anaconda.com/products/individual).
3. [Git](https://git-scm.com/downloads)

All the necessary dependencies can be found in the file `requirements.txt` or `environment.yml`.

To run the project:
1. Fork this repository.
2. Clone your repository.
          <pre>git clone <i>your_repository_address</i></pre>
3. Create conda environment.
          <pre>conda env create --name <i>name_of_environment</i> -f environment.yml </pre>
4. Activate your environment.
          <pre>conda activate <i>name_of_environment</i></pre>
5. Launch Jupyter Notebook.
          <pre>jupyter notebook</pre>
6. Enjoy!
