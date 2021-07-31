class ExplanationAlgorithmTemplate:
    @classmethod
    def find_counterfactual_multiple_k(cls, user_id, ks, model, data, args):
        """
        given a user, find an explanation for that user using the "pure FIA" algorithm
        Args:
            user_id: ID of user
            ks: a list of values of k to consider
            model: the recommender model, a Tensorflow Model object
            data: the dataset used to train the model, see dataset.py
            args: other args used for model

        Returns: a list explanations, each correspond to one value of k. Each explanation is a tuple consisting of:
                    - a set of items in the counterfactual explanation
                    - the originally recommended item
                    - a list of items in the original top k
                    - a list of predicted scores after the removal of the counterfactual explanation
                    - the predicted replacement item
        """
        raise Exception('Not implemented')
