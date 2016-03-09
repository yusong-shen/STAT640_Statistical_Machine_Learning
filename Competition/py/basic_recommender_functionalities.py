#!/usr/bin/env python
# -*- coding: utf-8 -*-


# coding: utf-8

## Basic Recommender Functionalities

# <h2>Basic Recommender Functionalities</h2>
# 
# The GraphLab Create recommender package implements a number of popular recommender models.  The models differ in how they make new predictions and recommendations given observed data, but they conform to the same API.  Namely, you can call `create()` to create the model, then call `score()`, `recommend()`, `evaluate()` on the returned model object.  
# 
# Let's walk through each of these functions in turn.
# 
# <h3>Creating a model</h3>
# 
# You can create a simple recommender model using `recommender.create()`.

# In[1]:

import graphlab
# Show graphs and sframes inside ipython notebook
graphlab.canvas.set_target('ipynb')


# In[2]:

# Load a small training set. The data is 2.8 MB.
training_data = graphlab.SFrame.read_csv("http://s3.amazonaws.com/dato-datasets/movie_ratings/training_data.csv",
                                         column_type_hints={"rating":int})

model = graphlab.recommender.create(training_data, user_id="user", item_id="movie")


# The above code automatically chose to create a `ItemSimilarityRecommender` based on the data provided.  We may directly create this type of model with our own chosen configuration for the training process, such as a particular choice of similarity. It defaults to `jaccard`, but you can set it to `cosine` if the data contains rating information.
# 

# In[3]:

model_cosine = graphlab.item_similarity_recommender.create(training_data, user_id="user", item_id="movie", target="rating",
                                                           similarity_type="cosine")


# Different models may have different configuration parameters and input argument semantics.  For example, `LinearRegressionModel` and `LogisticRegressionModel` treats additional columns in the training_data as side features to use for prediction.  See the <a href="https://dato.com/products/create/docs/">API reference</a> for details.  
# 
# <h3>Making predictions</h3>
# 
# `score()` makes rating predictions for any user-item query pair in the input SFrame (or Pandas dataframe).  For example, if the input contains the row "Alice, The Great Gatsby," `score()` will output a number that represents the model's prediction of how much "Alice" will like "The Great Gatsby."
# 
# The query data must have the same schema as the training data.  In other words, it must contain columns of the same name for user and item id, and they must be in the same column position, i.e., if the training data contains the user IDs in the second column, then it must be in the second column in the query data as well.  All other columns in the input query data are ignored.  (Exceptions are the `LinearRegressionModel` and `LogisticRegressionModel`, which treat the additional columns as side features.  See the <a href="https://dato.com/products/create/docs/">API documentation</a> for details.)  In this example, even though the input data contains the groundtruth ratings, they are ignored by the model.

# In[4]:

# Load some data for querying. The data is < 1MB.
# The query data must have the same columns ("user_id" and "item_id") as the training data.
# The column indices should also be the same, i.e., if the "user_id" column is the second
# column in the training data, then it also needs to be the second column in the query data.
query_data = graphlab.SFrame.read_csv("http://s3.amazonaws.com/dato-datasets/movie_ratings/query_data.csv", column_type_hints={"rating":int})
query_data.show()


# The output of `score()` is an SArray with as many rows as the query data.  The i-th entry in this SArray is the rating prediction for the i-th user-item pair in the query data.
# 
# The prediction scores are <i>unnormalized</i>, meaning that they may not conform to the scale of the original ratings given in the training data.  (Remember that this model was not trained to predict target ratings, but simply to rank items and make recommendations.  It can still make ratings predictions, but the scores you see may not map to the original scale.)

# In[5]:

# Now make some predictions 
query_result = model.predict(query_data)
query_result.head()


# With a sprinkle of SArray statistical magic dust, you can scale the model's output prediction scores to the scale of the original scores.

# In[6]:

# Scale the results to be on the same scale as the original ratings
scaled_result = (query_result - query_result.min())/(query_result.max() - query_result.min()) * query_data['rating'].max()
scaled_result.head()


# <h3>Making recommendations</h3>
# 
# Unlike `score()`, which returns raw predicted scores of user-item pairs, `recommend()` returns a ranked list of items.  The input parameter `users` should be an SArray of user ID for which to make recommendations.  If `users` is set to `None`, then `recommend()` makes recommendations for all users seen during training, while automatically excluding the items that the user has already rated in the training data. In other words, if the training data contains a row "Alice, The Great Gatsby", then `recommend()` will not recommend "The Great Gatsby" for user "Alice".  It will return at most `k` new items for each user, sorted by their rank.  It will return fewer than `k` items if there are not enough items that the user has not already rated or seen.
# 
# The output of recommend is an SFrame of four columns: "user", "item", "score", and "rank".

# In[7]:

recommend_result = model.recommend(users=None, k=5)
recommend_result.head()


# The raw scores for ItemSimilarityModel with Jaccard similarity is rather meaningless: a higher score could mean that the user rated more items, or the item in question is more popular, or it is very similar to many other items.  So ignore the score column in this instance.  The rank is what counts.

# <h3>Training and validation</h3>
# 
# If you've spent any time near machine learning nerds, you may have heard strange phrases like "hold-out validation" and "generalization error."  What on earth do they mean?  Basically, it has to do with measuring the accuracy of the model.  Measuring accuracy on training data is cheating, because the model already knows about that dataset and can easily output good predictions to fool you into believing that it's doing well.  Hence, it's more fair to evaluate the model on data that has *not* been seen by the model during training.  This is what the <i>validation dataset</i> is for.  It is common practice to divide the whole dataset into two parts, with one part being "held out" from the training process and used only for validation.
# 
# GraphLab Create allows you to do this with ease.  `recommender.util.random_split_by_user()` allows you to hold out a random subset of items for a number of users to be used as validation data.  Note that this is not the same as tossing a coin for each observation to decide whether it should be used as training or validation--a method that does not guarantee that *some* items are retained for each user for training.
# 

# In[8]:

training_subset, validation_subset = graphlab.recommender.util.random_split_by_user(training_data,
                                                                                    user_id="user", item_id="movie",
                                                                                    max_num_users=100, item_test_proportion=0.3) 


# In[9]:

model = graphlab.recommender.create(training_subset, user_id="user", item_id="movie", target="rating")


# <h3>Evaluation</h3>
# 
# To evaluate the accuracy or ranking performance of a recommender model, call `evaluate()` with a test or validation dataset.  The dataset should have the same schema (column headers and positions) as what was used during training.  If the model was trained to predict a target column of ratings, `evaluate()` will call `score()` to predict ratings for all the user-item pairs given in the testset, and output the resulting RMSE (Root-Mean-Squared Error) scores.  If the model was trained without a target column, `evaluate()` will rank the items for each user in the given test set, and evaluate the precision and recall using the given items.  In either case, the model outputs are evaluated against the groundtruth data provided in the input dataset.
# 
# The precision and recall scores are computed for different cutoff values (i.e., how many recommendations to take when computing the scores).  See the <a href="https://dato.com/products/create/docs/">API documentation</a> for more details.

# In[10]:

# Manually evaluate on the validation data. Evaluation results contain per-user, per-item and overall RMSEs.
rmse_results = model.evaluate(validation_subset)


# In[11]:

rmse_results['rmse_by_item'].show()


# In[12]:

rmse_results['rmse_by_user'].show()


# RMSE evaluation results show the user and the item with the best and worst average RMSE, as well as the overall average RMSE across all user-item pairs.

# In[13]:

# Now let's try a model that performs ranking by default, and look at the precision-recall validation results.
model2 = graphlab.ranking_factorization_recommender.create(training_subset, user_id="user", item_id="movie")
precision_recall_results = model2.evaluate(validation_subset)


# (Looking for more details about the modules and functions? Check out the <a href="https://dato.com/products/create/docs/">API docs</a>.)
