import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from recommenders.recommender import Recommender

from catboost import CatBoostRegressor

from collections import defaultdict


def prepare_users_df(interactions_df):
    # Make copy of interactions_df
    users_df = interactions_df.copy()

    # Get dummies
    users_df = pd.get_dummies(users_df, columns=[
        'term',
        'length_of_stay_bucket',
        'rate_plan',
        'room_segment',
        'n_people_bucket',
        'weekend_stay'
    ])

    # Drop column item_id
    users_df = users_df.drop(columns=['item_id'])

    # Group data by user_id using sum
    users_df = users_df.groupby('user_id').sum()

    # Dividing the numerical data in order to obtain the probability distribution of features for a given user
    users_df = users_df.div(users_df.sum(axis=1) / 6, axis=0)

    # Add prefix to columns
    users_df = users_df.add_prefix('user_')

    # Reset index
    users_df = users_df.reset_index()

    # Get list of user features
    user_features = users_df.columns.tolist()[1:]

    return users_df, user_features


def prepare_items_df(interactions_df):
    # Create copy of DataFrame
    items_df = interactions_df.copy()

    # Drop column user_id if present and drop duplicates
    if 'user_id' in items_df.columns:
        items_df = items_df.drop(columns=['user_id'])
        items_df = items_df.drop_duplicates()

    # Get dummies
    items_df = pd.get_dummies(items_df, columns=[
        'term',
        'length_of_stay_bucket',
        'rate_plan',
        'room_segment',
        'n_people_bucket',
        'weekend_stay'
    ], dtype=float)

    # Get list of item features
    item_features = items_df.columns.tolist()[1:]

    return items_df, item_features


def generateNegativeInteractions(interactions_df, seed=6789, n_neg_per_pos=5):
    # Set random seed
    rng = np.random.RandomState(seed=seed)

    # Get number of users and items
    n_users = np.max(interactions_df['user_id']) + 1
    n_items = np.max(interactions_df['item_id']) + 1

    # Generate interaction matrix
    r = np.zeros(shape=(n_users, n_items))
    for idx, interaction in interactions_df.iterrows():
        r[int(interaction['user_id'])][int(interaction['item_id'])] = 1

    # Generate negative interactions

    negative_interactions = []

    i = 0
    while i < n_neg_per_pos * len(interactions_df):
        sample_size = 1000
        user_ids = rng.choice(np.arange(n_users), size=sample_size)
        item_ids = rng.choice(np.arange(n_items), size=sample_size)

        j = 0
        while j < sample_size and i < n_neg_per_pos * len(interactions_df):
            if r[user_ids[j]][item_ids[j]] == 0:
                negative_interactions.append([user_ids[j], item_ids[j], 0])
                i += 1
            j += 1

    return negative_interactions


class ContentBasedUserItemRecommender(Recommender):
    """
    Linear recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, nn=False, cat=False, epochs=10):
        """
        Initialize base recommender params and variables.
        """
        self.model = LinearRegression()
        self.n_neg_per_pos = n_neg_per_pos

        self.recommender_df = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
        self.users_df = None
        self.users_dict = None
        self.user_features = None

        self.nn = nn
        self.cat = cat
        self.epochs = epochs

        self.uses_dot_product = True

        self.seed = seed
        self.rng = np.random.RandomState(seed=seed)

    def fit(self, interactions_df, users_df, items_df):
        """
        Training of the recommender.

        :param pd.DataFrame interactions_df: DataFrame with recorded interactions between users and items
            defined by user_id, item_id and features of the interaction.
        :param pd.DataFrame users_df: DataFrame with users and their features defined by user_id and the user feature columns.
        :param pd.DataFrame items_df: DataFrame with items and their features defined by item_id and the item feature columns.
        """

        interactions_df = interactions_df.copy()

        user_ids = interactions_df['user_id'].unique().tolist()
        item_ids = interactions_df['item_id'].unique().tolist()

        # Prepare users_df and items_df

        users_df, user_features = prepare_users_df(interactions_df)

        self.users_df = users_df

        self.users_dict = users_df.copy().set_index('user_id').to_dict('index')

        self.user_features = user_features

        items_df, item_features = prepare_items_df(interactions_df)
        items_df = items_df.loc[:, ['item_id'] + item_features]

        # Generate negative interactions

        negative_interactions = generateNegativeInteractions(interactions_df, seed=self.seed,
                                                             n_neg_per_pos=self.n_neg_per_pos)

        interactions_df = interactions_df.loc[:, ['user_id', 'item_id']]

        interactions_df.loc[:, 'interacted'] = 1

        interactions_df = pd.concat(
            [interactions_df, pd.DataFrame(negative_interactions, columns=['user_id', 'item_id', 'interacted'])],
            ignore_index=True)

        # Get the input data for the model

        interactions_df = pd.merge(interactions_df, users_df, on=['user_id'])
        interactions_df = pd.merge(interactions_df, items_df, on=['item_id'])

        if self.uses_dot_product:
            interactions_df[user_features] = interactions_df[user_features] \
                                             * interactions_df[item_features].values
            x = interactions_df.loc[:, user_features].values
        else:
            x = interactions_df.loc[:, user_features + item_features].values

        y = interactions_df['interacted'].values

        if self.cat:
            self.model.fit(x, y, silent=True, verbose=False)
        elif self.nn:
            self.model.fit(x, y, epochs=self.epochs, verbose=0)
        else:
            self.model.fit(x, y)

    def recommend(self, users_df, items_df, n_recommendations=1):
        """
        Serving of recommendations. Scores items in items_df for each user in users_df and returns
        top n_recommendations for each user.

        :param pd.DataFrame users_df: DataFrame with users and their features for which recommendations should be generated.
        :param pd.DataFrame items_df: DataFrame with items and their features which should be scored.
        :param int n_recommendations: Number of recommendations to be returned for each user.
        :return: DataFrame with user_id, item_id and score as columns returning n_recommendations top recommendations
            for each user.
        :rtype: pd.DataFrame
        """

        # Clean previous recommendations (iloc could be used alternatively)
        self.recommender_df = self.recommender_df[:0]

        item_df = items_df.copy()

        item_df, item_features = prepare_items_df(item_df)

        avg_user = self.users_df.copy().drop(columns='user_id').mean().to_frame().T

        # Score the items

        recommendations = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

        for ix, user in users_df.iterrows():
            if user['user_id'] in self.users_dict:
                user_df = pd.DataFrame.from_dict({user['user_id']: self.users_dict[user['user_id']]}, orient='index')
            else:
                user_df = avg_user.copy()

            input_df = item_df.copy()

            # Calculate scores

            if self.uses_dot_product:
                input_df[item_features] = item_df[item_features] * user_df.values
                if self.nn:
                    scores = self.model.predict(input_df.loc[:, item_features].values, verbose=0)
                    scores = scores.reshape(-1, )
                else:
                    scores = self.model.predict(input_df.loc[:, item_features].values)
            else:
                input_df = input_df.merge(user_df, how='cross')
                if self.nn:
                    scores = self.model.predict(input_df.loc[:, list(item_features) + self.user_features].values,
                                                verbose=0)
                    scores = scores.reshape(-1, )
                else:
                    scores = self.model.predict(input_df.loc[:, list(item_features) + self.user_features].values)

            # Get highest scores

            chosen_ids = np.argsort(-scores)[:n_recommendations]

            # Recommendations

            recommendations = []
            for item_id in chosen_ids:
                recommendations.append(
                    {
                        'user_id': user['user_id'],
                        'item_id': item_id,
                        'score': scores[item_id]
                    }
                )

            user_recommendations = pd.DataFrame(recommendations)

            self.recommender_df = pd.concat([self.recommender_df, user_recommendations])

        return self.recommender_df


class LinearRegressionCBUIRecommender(ContentBasedUserItemRecommender):
    """
    Linear regression recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        self.model = LinearRegression()


class SVRCBUIRecommender(ContentBasedUserItemRecommender):
    """
    SVR recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'kernel' in model_params:
            self.kernel = model_params['kernel']
        else:
            self.kernel = 'rbf'
        if 'C' in model_params:
            self.C = model_params['C']
        else:
            self.C = 1.0
        if 'epsilon' in model_params:
            self.epsilon = model_params['epsilon']
        else:
            self.epsilon = 0.1
        self.model = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon)


class RandomForestCBUIRecommender(ContentBasedUserItemRecommender):
    """
    Random forest recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'n_estimators' in model_params:
            self.n_estimators = int(model_params['n_estimators'])
        else:
            self.n_estimators = 100
        if 'max_depth' in model_params:
            self.max_depth = int(model_params['max_depth'])
        else:
            self.max_depth = 30
        if 'min_samples_split' in model_params:
            self.min_samples_split = int(model_params['min_samples_split'])
        else:
            self.min_samples_split = 30
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            random_state=111)


class XGBoostCBUIRecommender(ContentBasedUserItemRecommender):
    """
    XGBoost recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'n_estimators' in model_params:
            self.n_estimators = int(model_params['n_estimators'])
        else:
            self.n_estimators = 100
        if 'max_depth' in model_params:
            self.max_depth = int(model_params['max_depth'])
        else:
            self.max_depth = 30
        if 'min_samples_split' in model_params:
            self.min_samples_split = int(model_params['min_samples_split'])
        else:
            self.min_samples_split = 30
        if 'learning_rate' in model_params:
            self.learning_rate = model_params['learning_rate']
        else:
            self.learning_rate = 30
        self.model = GradientBoostingRegressor(
            n_estimators=self.n_estimators, max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            learning_rate=self.learning_rate, random_state=seed)


class CatBoostRegressorCBUIRecommender(ContentBasedUserItemRecommender):
    """
    CatBoost recommender class based on user and item features.
    """

    def __init__(self, seed=6789, n_neg_per_pos=5, cat=True, **model_params):
        """
        Initialize base recommender params and variables.
        """
        super().__init__(seed=seed, n_neg_per_pos=n_neg_per_pos)
        if 'learning_rate' in model_params:
            self.learning_rate = model_params['learning_rate']
        else:
            self.learning_rate = 0.01
        if 'iterations' in model_params:
            self.iterations = model_params['iterations']
        else:
            self.iterations = 1000
        if 'l2_leaf_reg' in model_params:
            self.l2_leaf_reg = model_params['l2_leaf_reg']
        else:
            self.l2_leaf_reg = 1.0
        if 'depth' in model_params:
            self.depth = model_params['depth']
        else:
            self.depth = 5

        self.model = CatBoostRegressor(
            loss_function='RMSE',
            learning_rate=self.learning_rate,
            iterations=self.iterations,
            l2_leaf_reg=self.l2_leaf_reg,
            depth=self.depth,
            verbose=False
        )
