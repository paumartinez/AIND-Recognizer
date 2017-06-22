import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        
        #Bayesian information criteria: BIC = -2 * logL + p * logN
        #where L is the likelihood of the fitted model, p is the number of parameters, 
        #and N is the number of data points. The term −2 log L decreases with
        #increasing model complexity (more parameters), whereas the penalties 2p or
        #p log N increase with increasing complexity. The BIC applies a larger penalty
        #when N > e2 = 7.4.

        #From https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/12
        #Initial state occupation probabilities = numStates
        #Transition probabilities = numStates*(numStates - 1)
        #Emission probabilities = numStates*numFeatures*2 = numMeans+numCovars

        #Then the total number of parameters are:
        #Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities


        best_score = float("-inf")
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components + 1):

            try:
                new_model = self.base_model(n_components)
                logL = new_model.score(self.X, self.lengths)
                logN = np.log(self.X.shape[0])
                
                features = self.X.shape[1]

                #Parameters = Initial state occupation probabilities + Transition probabilities + Emission probabilities
                p = n_components + n_components * (n_components - 1) + n_components * features * 2
                
                BIC = -2 * logL + p * logN

                if BIC > best_score:
                    best_score = BIC
                    best_model = new_model

            except:
                pass

        return best_model



class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
      
        best_score = float('-inf')
        best_model = None

        for n_components in range(self.min_n_components, self.max_n_components+1):
            likelihood = float('-inf')
            anti_likelihood = []

            try:
                new_model = self.base_model(n_components)
                likelihood = new_model.score(self.X, self.lengths)
            except:
                pass

            for word in self.hwords:
                if word != self.this_word:
                    X, lengths = self.hwords[word]
                try:
                    anti_likelihood.append(new_model.score(X, lengths))
                except:
                    pass
            

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                # DIC = log(P(X(i)) - average(log(P(X(all but i))
                DIC = likelihood - np.average(anti_likelihood)
            
            if  DIC > best_score:
                best_score = DIC
                best_model = new_model

        return best_model




class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV


        best_score = float("-inf")
        best_model = None

        n_splits = 3
        if len(self.sequences) < 2:
            return None
        elif len(self.sequences) == 2:
            n_splits = 2

        
        for n_components in range(self.min_n_components, self.max_n_components + 1):
            
            split_method = KFold(n_splits=n_splits)
            scores = []
            logL = []

            for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
             
                self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                X_test,  lengths_test  = combine_sequences(cv_test_idx, self.sequences)
                model = self.base_model(n_components)

                try:
                    logL = model.score(X_test, lengths_test)
                    scores.append(logL)
                
                except:
                    pass
            
            if len(scores) > 0:
                new_score = np.mean(scores)
            else:
                new_score = float("-inf")

            if new_score > best_score:
                best_score = new_score
                best_model = model

        return best_model

